# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from Modulus repo Copyright (c) NVIDIA CORPORATION & AFFILIATES

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Optional
import einops

import torch
from torch import Tensor

try:
    from typing import Self
except ImportError:
    # for Python versions < 3.11
    from typing_extensions import Self

import sys
sys.path.append("")
from models.graphcast.gnn_layers.embedder import (
    GraphCastDecoderEmbedder,
    GraphCastEncoderEmbedder,
)
from models.graphcast.gnn_layers.mesh_graph_decoder import MeshGraphDecoder
from models.graphcast.gnn_layers.mesh_graph_encoder import MeshGraphEncoder
from models.graphcast.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from models.graphcast.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn
from models.graphcast.utils.activations import get_activation
from models.graphcast.utils.meta import ModelMetaData
from models.graphcast.utils.module import Module
from models.graphcast.utils.graph import Graph

from models.graphcast.graph_cast_processor import GraphCastProcessor


@dataclass
class MetaData(ModelMetaData):
    name: str = "GraphCastNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class GraphCastNet(Module):
    """GraphCast network architecture

    Parameters
    ----------
    meshgraph_path : str
        Path to the meshgraph file. If not provided, the meshgraph will be created
        using PyMesh.
    static_dataset_path : str
        Path to the static dataset file.
    input_height: int
        Height (latitude) of the input
    input_width: int
        Width (longitude) of the input
    input_dim_grid_nodes : int, optional
        Input dimensionality of the grid node features, by default 474
    input_dim_mesh_nodes : int, optional
        Input dimensionality of the mesh node features, by default 3
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim_grid_nodes : int, optional
        Final output dimensionality of the edge features, by default 227
    processor_layers : int, optional
        Number of processor layers, by default 16
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    activation_fn : str, optional
        Type of activation function, by default "silu"
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    use_cugraphops_encoder : bool, default=False
        Flag to select cugraphops kernels in encoder
    use_cugraphops_processor : bool, default=False
        Flag to select cugraphops kernels in the processor
    use_cugraphops_decoder : bool, default=False
        Flag to select cugraphops kernels in the decoder
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    partition_size : int, default=1
        Number of process groups across which graphs are distributed. If equal to 1,
        the model is run in a normal Single-GPU configuration.
    partition_group_name : str, default=None
        Name of process group across which graphs are distributed. If partition_size
        is set to 1, the model is run in a normal Single-GPU configuration and the
        specification of a process group is not necessary. If partitition_size > 1,
        passing no process group name leads to a parallelism across the default
        process group. Otherwise, the group size of a process group is expected
        to match partition_size.
    expect_partitioned_input : bool, default=False,
        Flag indicating whether the model expects the input to be already
        partitioned. This can be helpful e.g. in multi-step rollouts to avoid
        aggregating the output just to distribute it in the next step again.
    produce_aggregated_output : bool, default=True,
        Flag indicating whether the model produces the aggregated output on each
        rank of the progress group across which the graph is distributed or
        whether the output is kept distributed. This can be helpful e.g.
        in multi-step rollouts to avoid aggregating the output just to distribute
        it in the next step again.

    Note
    ----
    Based on these papers:
    - "GraphCast: Learning skillful medium-range global weather forecasting"
        https://arxiv.org/abs/2212.12794
    - "Forecasting Global Weather with Graph Neural Networks"
        https://arxiv.org/abs/2202.07575
    - "Learning Mesh-Based Simulation with Graph Networks"
        https://arxiv.org/abs/2010.03409
    - "MultiScale MeshGraphNets"
        https://arxiv.org/abs/2210.00612
    """

    def __init__(
        self,
        meshgraph_path: str,
        input_height: int = 721,
        input_width: int = 1440,
        constant_channels: int = 4,
        prescribed_channels: int = 1,
        prognostic_channels: int = 8,
        #input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        #output_dim_grid_nodes: int = 227,
        processor_layers: int = 16,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        use_cugraphops_encoder: bool = False,
        use_cugraphops_processor: bool = False,
        use_cugraphops_decoder: bool = False,
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
        partition_size: int = 1,
        partition_group_name: Optional[str] = None,
        expect_partitioned_input: bool = False,
        produce_aggregated_output: bool = True,
        context_size: int = 1,
        **kwargs
    ):
        super().__init__(meta=MetaData())

        self.context_size = context_size
        input_dim_grid_nodes = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        output_dim_grid_nodes = prognostic_channels

        self.is_distributed = False
        if partition_size > 1:
            self.is_distributed = True
        self.expect_partitioned_input = expect_partitioned_input
        self.produce_aggregated_output = produce_aggregated_output

        # create the lat_lon_grid
        self.latitudes = torch.linspace(-90, 90, steps=input_height)
        self.longitudes = torch.linspace(-180, 180, steps=input_width + 1)[1:]
        self.lat_lon_grid = torch.stack(
            torch.meshgrid(self.latitudes, self.longitudes, indexing="ij"), dim=-1
        )
        self.has_static_data = constant_channels > 0

        # Set activation function
        activation_fn = get_activation(activation_fn)

        # construct the graph
        try:
            self.graph = Graph(meshgraph_path, self.lat_lon_grid)
        except FileNotFoundError:
            raise FileNotFoundError(
                "The icospheres_path is corrupted. "
                "Tried using pymesh to generate the graph but could not find pymesh"
            )
        self.mesh_graph = self.graph.create_mesh_graph(verbose=False)  # Processor
        self.g2m_graph = self.graph.create_g2m_graph(verbose=False)  # Translates from LatLon to Icospheres
        self.m2g_graph = self.graph.create_m2g_graph(verbose=False)  # Translates form Icospheres to LatLon

        self.g2m_edata = self.g2m_graph.edata["x"]
        self.m2g_edata = self.m2g_graph.edata["x"]
        self.mesh_edata = self.mesh_graph.edata["x"]
        self.mesh_ndata = self.mesh_graph.ndata["x"]

        if use_cugraphops_encoder or self.is_distributed:
            self.g2m_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.g2m_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
            )
            self.g2m_edata = self.g2m_edata[edge_perm]

            if self.is_distributed:
                self.g2m_edata = self.g2m_graph.get_edge_features_in_partition(
                    self.g2m_edata
                )

        if use_cugraphops_decoder or self.is_distributed:
            self.m2g_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.m2g_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
            )
            self.m2g_edata = self.m2g_edata[edge_perm]

            if self.is_distributed:
                self.m2g_edata = self.m2g_graph.get_edge_features_in_partition(
                    self.m2g_edata
                )

        if use_cugraphops_processor or self.is_distributed:
            self.mesh_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.mesh_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
            )
            self.mesh_edata = self.mesh_edata[edge_perm]
            if self.is_distributed:
                self.mesh_edata = self.mesh_graph.get_edge_features_in_partition(
                    self.mesh_edata
                )
                self.mesh_ndata = self.mesh_graph.get_dst_node_features_in_partition(
                    self.mesh_ndata
                )

        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.output_dim_grid_nodes = output_dim_grid_nodes
        self.input_res = (input_height, input_width)

        # by default: don't checkpoint at all
        self.model_checkpoint_fn = set_checkpoint_fn(False)
        self.encoder_checkpoint_fn = set_checkpoint_fn(False)
        self.decoder_checkpoint_fn = set_checkpoint_fn(False)

        # initial feature embedder
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )
        self.decoder_embedder = GraphCastDecoderEmbedder(
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # grid2mesh encoder
        self.encoder = MeshGraphEncoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_src_nodes=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # icosahedron processor
        if processor_layers <= 2:
            raise ValueError("Expected at least 3 processor layers")
        self.processor_encoder = GraphCastProcessor(
            aggregation=aggregation,
            processor_layers=1,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )
        self.processor = GraphCastProcessor(
            aggregation=aggregation,
            processor_layers=processor_layers - 2,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )
        self.processor_decoder = GraphCastProcessor(
            aggregation=aggregation,
            processor_layers=1,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # mesh2grid decoder
        self.decoder = MeshGraphDecoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # final MLP
        self.finale = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim_grid_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

    def set_checkpoint_model(self, checkpoint_flag: bool):
        """Sets checkpoint function for the entire model.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. In this case, all the other gradient checkpoitings
        will be disabled. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        # force a single checkpoint for the whole model
        self.model_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)
        if checkpoint_flag:
            self.processor.set_checkpoint_segments(-1)
            self.encoder_checkpoint_fn = set_checkpoint_fn(False)
            self.decoder_checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_processor(self, checkpoint_segments: int):
        """Sets checkpoint function for the processor excluding the first and last
        layers.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_segments` flag. If `checkpoint_segments` is positive,
         the function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`, with number of checkpointing segments equal to
        `checkpoint_segments`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_segments : int
            Number of checkpointing segments for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.processor.set_checkpoint_segments(checkpoint_segments)

    def set_checkpoint_encoder(self, checkpoint_flag: bool):
        """Sets checkpoint function for the embedder, encoder, and the first of
        the processor.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.encoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def set_checkpoint_decoder(self, checkpoint_flag: bool):
        """Sets checkpoint function for the last layer of the processor, the decoder,
        and the final MLP.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.decoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def encoder_forward(
        self,
        grid_nfeat: Tensor,
    ) -> Tensor:
        """Forward method for the embedder, encoder, and the first of the processor.

        Parameters
        ----------
        grid_nfeat : Tensor
            Node features for the latitude-longitude grid.

        Returns
        -------
        mesh_efeat_processed: Tensor
            Processed edge features for the multimesh.
        mesh_nfeat_processed: Tensor
            Processed node features for the multimesh.
        grid_nfeat_encoded: Tensor
            Encoded node features for the latitude-longitude grid.
        """

        # embedd graph features
        (
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            g2m_efeat_embedded,
            mesh_efeat_embedded,
        ) = self.encoder_embedder(
            grid_nfeat,
            self.mesh_ndata,
            self.g2m_edata,
            self.mesh_edata,
        )

        # encode lat/lon to multimesh
        grid_nfeat_encoded, mesh_nfeat_encoded = self.encoder(
            g2m_efeat_embedded,
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            self.g2m_graph,
        )

        # process multimesh graph
        mesh_efeat_processed, mesh_nfeat_processed = self.processor_encoder(
            mesh_efeat_embedded,
            mesh_nfeat_encoded,
            self.mesh_graph,
        )

        return mesh_efeat_processed, mesh_nfeat_processed, grid_nfeat_encoded

    def decoder_forward(
        self,
        mesh_efeat_processed: Tensor,
        mesh_nfeat_processed: Tensor,
        grid_nfeat_encoded: Tensor,
    ) -> Tensor:
        """Forward method for the last layer of the processor, the decoder,
        and the final MLP.

        Parameters
        ----------
        mesh_efeat_processed : Tensor
            Multimesh edge features processed by the processor.
        mesh_nfeat_processed : Tensor
            Multi-mesh node features processed by the processor.
        grid_nfeat_encoded : Tensor
            The encoded node features for the latitude-longitude grid.

        Returns
        -------
        grid_nfeat_finale: Tensor
            The final node features for the latitude-longitude grid.
        """

        # process multimesh graph
        _, mesh_nfeat_processed = self.processor_decoder(
            mesh_efeat_processed,
            mesh_nfeat_processed,
            self.mesh_graph,
        )

        m2g_efeat_embedded = self.decoder_embedder(self.m2g_edata)

        # decode multimesh to lat/lon
        grid_nfeat_decoded = self.decoder(
            m2g_efeat_embedded, grid_nfeat_encoded, mesh_nfeat_processed, self.m2g_graph
        )

        # map to the target output dimension
        grid_nfeat_finale = self.finale(
            grid_nfeat_decoded,
        )

        return grid_nfeat_finale

    def custom_forward(self, grid_nfeat: Tensor) -> Tensor:
        """GraphCast forward method with support for gradient checkpointing.

        Parameters
        ----------
        grid_nfeat : Tensor
            Node features of the latitude-longitude graph.

        Returns
        -------
        grid_nfeat_finale: Tensor
            Predicted node features of the latitude-longitude graph.
        """
        (
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
        ) = self.encoder_checkpoint_fn(
            self.encoder_forward,
            grid_nfeat,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        # checkpoint of processor done in processor itself
        mesh_efeat_processed, mesh_nfeat_processed = self.processor(
            mesh_efeat_processed,
            mesh_nfeat_processed,
            self.mesh_graph,
        )

        grid_nfeat_finale = self.decoder_checkpoint_fn(
            self.decoder_forward,
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        return grid_nfeat_finale

    def forward(
        self,
        constants: torch.Tensor = None,
        prescribed: torch.Tensor = None,
        prognostic: torch.Tensor = None
    ) -> torch.Tensor:
        """
        ...
        """
        # constants: [B, 1, C, H, W]
        # prescribed: [B, T, C, H, W]
        # prognostic: [B, T, C, H, W]
        outs = []

        for t in range(self.context_size, prognostic.shape[1]):
            
            t_start = max(0, t-(self.context_size))
            if t == self.context_size:
                # Initial condition
                prognostic_t = prognostic[:, t_start:t]
                x_t = self.prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t_start:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )
            else:
                # In case of context_size > 1, blend prognostic input with outputs from previous time steps
                prognostic_t = torch.cat(
                    tensors=[prognostic[:, t_start:self.context_size],        # Prognostic input before context_size
                             torch.stack(outs, dim=1)[:, -self.context_size:]].to(device=prognostic.device),  # Outputs since context_size
                    dim=1
                )
                x_t = self.prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )

            # Forward input through model
            out = prognostic_t[:, -1] + self.forward_one_step(invar=x_t)
            outs.append(out.cpu())
        
        return torch.stack(outs, dim=1)

    def forward_one_step(
        self,
        invar: Tensor
    ) -> Tensor:
        outvar = self.model_checkpoint_fn(
            self.custom_forward,
            invar,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        return self.prepare_output(outvar, self.produce_aggregated_output)

    def prepare_inputs(
        self,
        constants: Tensor,
        prescribed: Tensor,
        prognostic: Tensor
    ) -> Tensor:
        """Prepares the input to the model in the required shape.

        Parameters
        ----------
        invar : Tensor
            Input in the shape [N, T, C, H, W].

        expect_partitioned_input : bool
            flag indicating whether input is partioned according to graph partitioning scheme

        Returns
        -------
        Tensor
            Reshaped input.
        """
        invar = []
        if prescribed is not None: invar.append(einops.rearrange(prescribed, "b t c h w -> b (t c) h w"))
        if prognostic is not None: invar.append(einops.rearrange(prognostic, "b t c h w -> b (t c) h w"))
        if constants is not None: invar.append(constants[:, 0])
        invar = torch.cat(invar, dim=1)
        
        if invar.size(0) != 1:
            raise ValueError("GraphCast does not support batch size > 1")
        
        invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)
        if self.is_distributed:
            # partition node features
            invar = self.g2m_graph.get_src_node_features_in_partition(invar)
        
        return invar

    def prepare_output(self, outvar: Tensor, produce_aggregated_output: bool) -> Tensor:
        """Prepares the output of the model in the shape [N, C, H, W].

        Parameters
        ----------
        outvar : Tensor
            Output of the final MLP of the model.

        produce_aggregated_output : bool
            flag indicating whether output is gathered onto each rank
            or kept distributed

        Returns
        -------
        Tensor
            The reshaped output of the model.
        """
        if produce_aggregated_output or not self.is_distributed:
            # default case: output of shape [N, C, H, W]
            if self.is_distributed:
                outvar = self.m2g_graph.get_global_dst_node_features(outvar)

            outvar = outvar.permute(1, 0)
            outvar = outvar.view(self.output_dim_grid_nodes, *self.input_res)
            outvar = torch.unsqueeze(outvar, dim=0)

        else:
            # keep partition of H, W, i.e. produce [N, C, P]
            outvar = outvar.permute(1, 0).unsqueeze(dim=0)

        return outvar

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Moves the object to the specified device, dtype, or format.
        This method moves the object and its underlying graph and graph features to
        the specified device, dtype, or format, and returns the updated object.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `torch._C._nn._parse_to` function.
        **kwargs : Any
            Keyword arguments to be passed to the `torch._C._nn._parse_to` function.

        Returns
        -------
        GraphCastNet
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super(GraphCastNet, self).to(*args, **kwargs)

        self.g2m_edata = self.g2m_edata.to(*args, **kwargs)
        self.m2g_edata = self.m2g_edata.to(*args, **kwargs)
        self.mesh_ndata = self.mesh_ndata.to(*args, **kwargs)
        self.mesh_edata = self.mesh_edata.to(*args, **kwargs)

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.g2m_graph = self.g2m_graph.to(device)
        self.mesh_graph = self.mesh_graph.to(device)
        self.m2g_graph = self.m2g_graph.to(device)

        return self
