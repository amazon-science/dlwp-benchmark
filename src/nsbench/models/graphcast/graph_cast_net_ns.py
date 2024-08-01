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
from typing import Any, Optional, Callable, Union, List, Tuple
from itertools import chain

import torch
from torch import Tensor
import numpy as np
import networkx as nx
import einops

try:
    import dgl  # noqa: F401 for docs
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    from typing import Self
except ImportError:
    # for Python versions < 3.11
    from typing_extensions import Self

import sys
sys.path.append("")
from models.graphcast.gnn_layers.mesh_edge_block import MeshEdgeBlock
from models.graphcast.gnn_layers.mesh_node_block import MeshNodeBlock
from models.graphcast.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from models.graphcast.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn
from models.graphcast.utils.activations import get_activation
from models.graphcast.utils.meta import ModelMetaData
from models.graphcast.utils.module import Module


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


class GraphCastNetNS(Module):
    """GraphCast network architecture for Navier-Stokes data on a quadratic shape, i.e., [p, p]

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
        input_height: int = 32,
        input_width: int = 32,
        downscale_factor: int = None,
        context_size: int = 1,
        nhop_neighbors: List = [2],
        input_dim_nodes: int = 1,
        input_dim_edges: int = 3,
        output_dim: int = 1,
        processor_layers: int = 16,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 32,
        hidden_dim_node_encoder: int = 32,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 32,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 32,
        num_layers_node_decoder: int = 2,
        aggregation: str = "sum",
        activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        recompute_activation: bool = False,
        expect_partitioned_input: bool = False,
        produce_aggregated_output: bool = True,
        device = torch.device("cpu"),
        **kwargs
    ):
        super().__init__(meta=MetaData())

        input_height = input_height//downscale_factor
        input_width = input_width//downscale_factor

        self.expect_partitioned_input = expect_partitioned_input
        self.produce_aggregated_output = produce_aggregated_output
        self.context_size = context_size
        input_dim_nodes = input_dim_nodes * context_size

        # Set activation function
        activation_fn = get_activation(activation_fn)

        # construct the graph
        self.mesh_graph = self.create_grid_2d_graph(
            height=input_height,
            width=input_width,
            nhop_neighbors=nhop_neighbors
        ).to(device=device)
        self.efeats = self.create_edge_features(
            height=input_height,
            width=input_width,
            nhop_neighbors=nhop_neighbors
        ).to(device=device)

        self.node_encoder = MeshGraphMLP(
            input_dim=input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation
        )
        self.edge_encoder = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_layers,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )
        self.node_decoder = MeshGraphMLP(
            input_dim=hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

        self.perm_idcs = torch.randperm(64*64).to(device=device)
        self.perm_idcs_inverse = torch.zeros_like(self.perm_idcs)
        self.perm_idcs_inverse[self.perm_idcs] = torch.arange(len(self.perm_idcs)).to(device=device)

    def create_grid_2d_graph(
        self,
        height: int = 64,
        width: int = 64,
        periodic: bool = True,
        nhop_neighbors: List = [2]
    ) -> dgl.graph:
        # Create 1-hop 2D graph
        graph = nx.grid_2d_graph(
            m=height,
            n=width,
            periodic=periodic
        )
        self.graph_nx = dgl.to_networkx(dgl.to_bidirected(dgl.from_networkx(graph)))  # For distance calculation later

        # Iterate over all nodes and find n-hop neighbors. Store all edges to later add them to the graph
        nhop_neighbors = np.array(nhop_neighbors)
        max_nhop = max(nhop_neighbors)
        new_edges = []
        for node in graph.nodes:
            # Only consider nodes that have a coordinate listed in n-hop distance
            if not ((node[0]%nhop_neighbors == 0).any() and ((node[1]%nhop_neighbors == 0).any())): continue
            # Determine all neighbors that are in the considered n-hop distances and perpendicular to the node
            cutoff = max_nhop - max(max(node)%nhop_neighbors)  # Size of neighborhood depends on position in grid
            neighbors = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff)
            neighbors = {
                v: neighbors[v] for v in neighbors
                if (neighbors[v] in nhop_neighbors)  # neighbor has relevant n-hop distance
                and (v[0] == node[0] or v[1] == node[1])  # perpendicular (and not diagonal) neighbor
            }
            # Store the respective edges to add them to the graph later
            new_edges = new_edges + [[node, neighbor] for neighbor in neighbors]
        graph.add_edges_from(new_edges)

        return dgl.to_bidirected(dgl.from_networkx(graph))

    def create_edge_features(
        self,
        height: int = 64,
        width: int = 64,
        nhop_neighbors: List = [2]
    ) -> Tensor:
        max_dist = max(nhop_neighbors)
        # Get edges in OOD(?) format, iterate over all edges and compute edge features ([dir_y, dir_x, dist])
        efrom, eto = self.mesh_graph.edges()
        edge_features = []
        for u_idx, u in enumerate(efrom):
            v = eto[u_idx]
            uxy, vxy = torch.tensor([u//height, u%width]), torch.tensor([v//height, v%width])
            normal = vxy - uxy
            # Consider periodic cases, e.g., when left-most nodes have an edge to right-most nodes
            normal[normal>=height-1-max_dist] = -1
            normal[normal>=width-1-max_dist] = -1
            normal[normal<=-(height-1-max_dist)] = 1
            normal[normal<=-(width-1-max_dist)] = 1
            # Calculate (normalized) distance between nodes, clip the normal to [-1, 1] and create edge feature
            dist = torch.tensor(nx.shortest_path_length(G=self.graph_nx, source=int(u), target=int(v)) / max_dist)
            normal = normal.clip(min=-1, max=1)
            edge_feature = torch.cat([normal, dist.unsqueeze(0)])
            edge_features.append(edge_feature)
        return torch.stack(edge_features).type(torch.float32)

    def update_nodes_and_edges(
        self,
        nfeat: Tensor,
        efeat: Tensor
    ) -> Tensor:
        nfeat_enc = self.node_encoder(nfeat)
        efeat_enc = self.edge_encoder(efeat)
        nfeat_processed = self.processor(
            nfeat_enc,
            efeat_enc,
            self.mesh_graph,
        )
        # map to the target output dimension
        nfeat_finale = self.node_decoder(
            nfeat_processed,
        )
        return nfeat_finale

    def forward(
        self,
        x: Tensor,
        teacher_forcing_steps: int = 10
    ) -> Tensor:
        # x: [B, T, D, H, W]
        b, t, d, h, w = x.shape
        
        outs = []
        for t in range(x.shape[1]):
            
            # Prepare input, depending on teacher forcing or closed loop
            if t < teacher_forcing_steps:  # Teacher forcing: Feed observations into the model
                x_t = einops.rearrange(x[:, max(0, t-(self.context_size-1)):t+1], "b t d h w -> b (t d) h w")
            else:  # Closed loop: Feed output from previous time step as input into the model
                if self.context_size == 0:
                    x_t = out
                else:
                    # With teacher_forcing_steps=25 and context_size=5:
                    # t =  25 -> ts = 4, leading to cat([ x_t[21:25] , outs[-1:] ])
                    # t =  26 -> ts = 3, leading to cat([ x_t[22:25] , outs[-2:] ])
                    # ...
                    # t >= 30 -> ts = 0, leading to cat([ ---------- , outs[-5:] ])
                    ts = max(0, (teacher_forcing_steps - t - 1) + self.context_size)
                    x_obs = x[:, teacher_forcing_steps-ts:teacher_forcing_steps]
                    x_out = torch.stack(outs[-(self.context_size-ts):], dim=1)
                    x_t = einops.rearrange(torch.cat([x_obs, x_out], axis=1), "b t d h w -> b (t d) h w")
            
            # Forward input through model if context size has been reached
            if t < self.context_size - 1:
                # Not forwarding through model because context size not yet reached in output. Instead, consider the
                # most recent input as output.
                out = x_t[:, -1:]
            else:
                node_features = einops.rearrange(x_t, "b d h w -> (b h w) d")
                out = self.update_nodes_and_edges(
                    nfeat=node_features,
                    efeat=self.efeats
                )
                out = x_t[:, -1:] + einops.rearrange(out, "(b h w) d -> b d h w", b=b, h=h, w=w, d=d)
            
            outs.append(out)
        
        return torch.stack(outs, dim=1)



class MeshGraphNetProcessor(torch.nn.Module):
    """MeshGraphNet processor block"""

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: torch.nn.Module = torch.nn.SiLU(),
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments

        edge_block_invars = (
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_edge,
            activation_fn,
            norm_type,
            do_concat_trick,
            False,
        )
        node_block_invars = (
            aggregation,
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_node,
            activation_fn,
            norm_type,
            False,
        )

        edge_blocks = [
            MeshEdgeBlock(*edge_block_invars) for _ in range(self.processor_size)
        ]
        node_blocks = [
            MeshNodeBlock(*node_block_invars) for _ in range(self.processor_size)
        ]
        layers = list(chain(*zip(edge_blocks, node_blocks)))

        self.processor_layers = torch.nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        self.set_checkpoint_segments(self.num_processor_checkpoint_segments)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        """
        Set the number of checkpoint segments

        Parameters
        ----------
        checkpoint_segments : int
            number of checkpoint segments

        Raises
        ------
        ValueError
            if the number of processor layers is not a multiple of the number of
            checkpoint segments
        """
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def run_function(
        self, segment_start: int, segment_end: int
    ) -> Callable[
        [Tensor, Tensor, Union[DGLGraph, List[DGLGraph]]], Tuple[Tensor, Tensor]
    ]:
        """Custom forward for gradient checkpointing

        Parameters
        ----------
        segment_start : int
            Layer index as start of the segment
        segment_end : int
            Layer index as end of the segment

        Returns
        -------
        Callable
            Custom forward function
        """
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            edge_features: Tensor,
            graph: Union[DGLGraph, List[DGLGraph]],
        ) -> Tuple[Tensor, Tensor]:
            """Custom forward function"""
            for module in segment:
                edge_features, node_features = module(
                    edge_features, node_features, graph
                )
            return edge_features, node_features

        return custom_forward

    @torch.jit.unused
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
    ) -> Tensor:
        for segment_start, segment_end in self.checkpoint_segments:
            edge_features, node_features = self.checkpoint_fn(
                self.run_function(segment_start, segment_end),
                node_features,
                edge_features,
                graph,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        return node_features
