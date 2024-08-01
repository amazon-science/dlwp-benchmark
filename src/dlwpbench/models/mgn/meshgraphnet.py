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

import torch
import torch.nn as nn
import numpy as np
import scipy as sp
from torch import Tensor
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
from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple, Union

import sys
sys.path.append("")
from models.graphcast.gnn_layers.mesh_edge_block import MeshEdgeBlock
from models.graphcast.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from models.graphcast.gnn_layers.mesh_node_block import MeshNodeBlock
from models.graphcast.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn
from models.graphcast.utils.meta import ModelMetaData
from models.graphcast.utils.module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "MeshGraphNet"
    # Optimization, no JIT as DGLGraph causes trouble
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class MeshGraphNet(Module):
    """MeshGraphNet network architecture

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : int
        Number of edge features
    output_dim : int
        Number of outputs
    processor_size : int, optional
        Number of message passing blocks, by default 15
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : int, optional
        Number of MLP layers for the node feature encoder, by default 2
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : int, optional
        Number of MLP layers for the edge feature encoder, by default 2
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : int, optional
        Number of MLP layers for the node feature decoder, by default 2
    aggregation: str, optional
        Message aggregation type, by default "sum"
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    num_processor_checkpoint_segments: int, optional
        Number of processor segments for gradient checkpointing, by default 0 (checkpointing disabled)

    Example
    -------
    >>> model = modulus.models.meshgraphnet.MeshGraphNet(
    ...         input_dim_nodes=4,
    ...         input_dim_edges=3,
    ...         output_dim=2,
    ...     )
    >>> graph = dgl.rand_graph(10, 5)
    >>> node_features = torch.randn(10, 4)
    >>> edge_features = torch.randn(5, 3)
    >>> output = model(node_features, edge_features, graph)
    >>> output.size()
    torch.Size([10, 2])

    Note
    ----
    Reference: Pfaff, Tobias, et al. "Learning mesh-based simulation with graph networks."
    arXiv preprint arXiv:2010.03409 (2020).
    """

    def __init__(
        self,
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        input_dim_edges: int = 2,
        context_size: int = 5,
        processor_size: int = 15,
        message_passing_steps: int = 1,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        graph_type: str = "grid_2d",
        **kwargs
    ):
        super().__init__(meta=MetaData())

        input_dim_nodes = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        output_dim = prognostic_channels

        self.context_size = context_size
        self.message_passing_steps = message_passing_steps
        self.on_device = kwargs["device"]

        if graph_type == "grid_2d":
            self.create_grid_2d_graph(
                height=kwargs["graph"].height,
                width=kwargs["graph"].width,
                periodic=kwargs["graph"].periodic
            )
        elif graph_type == "delaunay":
            self.create_delaunay_graph(
                height=kwargs["graph"].height,
                width=kwargs["graph"].width,
                periodic=kwargs["graph"].periodic
            )
        elif graph_type == "grid_2d_8stencil":
            self.create_grid_2d_graph_8stencil(
                height=kwargs["graph"].height,
                width=kwargs["graph"].width,
                periodic=kwargs["graph"].periodic
            )
        else:
            raise ValueError(f"graph_type is '{graph_type}' but should be any of ['grid_2d', 'delaunay', "
                             f"'grid_2d_8stencil'].")
        # Assign data schemes to graph
        self.graph.ndata["x"] = torch.randn(self.graph.num_nodes(), input_dim_nodes).to(device=self.on_device)
        self.graph.edata["x"] = torch.randn(self.graph.num_edges(), input_dim_edges).to(device=self.on_device)
        self.graph.ndata["y"] = torch.randn(self.graph.num_nodes(), output_dim).to(device=self.on_device)

        self.update_batched_graph(batch_size=1)
        
        self.edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=nn.ReLU(),
            norm_type=None,
            recompute_activation=False,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=nn.ReLU(),
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )

    def update_batched_graph(
        self,
        batch_size
    ) -> None:
        self.batched_graph = dgl.batch([self.graph for b in range(batch_size)]).to(device=self.on_device)
        self.batched_edge_features = einops.rearrange(
            self.edge_features.unsqueeze(0).expand(batch_size, -1, -1), "b e d -> (b e) d"
        ).to(device=self.on_device)

    def create_grid_2d_graph(
        self,
        height: int = 64,
        width: int = 64,
        periodic: tuple = (False, True)
    ):
        graph = nx.grid_2d_graph(
            m=height,
            n=width,
            periodic=periodic
        )
        self.graph = dgl.to_bidirected(dgl.from_networkx(graph)).to(device=self.on_device)
        # Create edge features for this graph
        self.create_edge_features(
            height=height,
            width=width
        )

    def create_grid_2d_graph_8stencil(
        self,
        height: int = 64,
        width: int = 64,
        periodic: tuple = (False, True)
    ):
        graph = nx.grid_2d_graph(
            m=height,
            n=width,
            periodic=periodic
        )

        # Iterate over all nodes and add diagonal 1-hop neighbors. Store all edges to later add them to the graph
        new_edges = []
        diagonals = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
        for node in graph.nodes:
            neighbors = (node + diagonals) % height
            #if periodic:
            #    neighbors[0] = neighbors[0] % height
            #    neighbors[1] = neighbors[1] % width
            #else:
            #    neighbors[0] = neighbors.clip(min=0, max=height)
            #    neighbors[1] = neighbors.clip(min=0, max=width)
            new_edges = new_edges + [[node, tuple(neighbor)] for neighbor in neighbors]
        graph.add_edges_from(new_edges)
        self.graph = dgl.to_bidirected(dgl.from_networkx(graph)).to(device=self.on_device)

        # Create edge features for this graph
        self.create_edge_features(height=height, width=width, add_distance=True)

    def create_delaunay_graph(
        self,
        height: int = 64,
        width: int = 64,
        periodic: bool = True
    ) -> dgl.graph:
        # Create 1-hop 2D graph
        x = np.arange(start=0, step=1, stop=width+1)
        #y = np.arange(start=0, step=1, stop=height+1)
        y = np.arange(start=0, step=1, stop=height)
        xx,yy = np.meshgrid(x,y)
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        # Triangulate vertices
        receiver_tri = sp.spatial.Delaunay(np.transpose(np.vstack((xx.flatten(), yy.flatten()))))
        simplices = receiver_tri.simplices

        # Set last column equal to 1st (which effectively closes the horizontal gap)
        if periodic:
            #for i in range(height+1):
            for i in range(height):
                simplices[simplices == (width+1) * i + width] = (width+1) * i
            # Set last row equal to 1st (close vertical gap) -- not desired in cylinder mesh!
            #for j in range(width+1):
            #    simplices[simplices == height * (width+1) + j] = j

        # Create graph by adding all cycles represented by the simplices
        graph = nx.Graph()
        for path in simplices:
            nx.add_cycle(graph, path)

        # Build bidirected DGL graph and create according edge features
        self.graph = dgl.to_bidirected(dgl.from_networkx(graph)).to(device=self.on_device)
        self.create_edge_features(height=height, width=width, add_distance=False)

    def create_edge_features(
        self,
        height: int = 64,
        width: int = 64,
        add_distance: bool = False
    ) -> Tensor:
        efrom, eto = self.graph.edges()
        edge_features = []
        if add_distance: max_dist = 0
        
        for u_idx, u in enumerate(efrom):
            v = eto[u_idx]
            uxy, vxy = torch.tensor([u//height, u%width]), torch.tensor([v//height, v%width])
            normal = vxy - uxy
            # Consider periodic cases, e.g., when left-most nodes have an edge to right-most nodes
            normal[normal==height-1] = -1
            normal[normal==width-1] = -1
            normal[normal==-(height-1)] = 1
            normal[normal==-(width-1)] = 1
            if add_distance:
                distance = normal.abs().sum().sqrt()
                edge_feature = torch.cat([normal, distance.unsqueeze(0)])
                max_dist = max(max_dist, distance)
            else:
                edge_feature = normal
            edge_features.append(edge_feature)
        
        self.edge_features = torch.stack(edge_features).type(torch.float32).to(device=self.on_device)
        if add_distance: self.edge_features[:, -1] = self.edge_features[:, -1] / max_dist  # Normalize distance

    def create_edge_features_norm_dist(
        self,
        height: int = 64,
        width: int = 64
    ) -> Tensor:
        efrom, eto = self.graph.edges()
        edge_features = []
        max_dist = 0
        for u_idx, u in enumerate(efrom):
            v = eto[u_idx]
            uxy, vxy = torch.tensor([u//height, u%width]), torch.tensor([v//height, v%width])
            normal = vxy - uxy
            # Consider periodic cases, e.g., when a left-most node has an edge to a right-most node
            normal[normal==height-1] = -1
            normal[normal==width-1] = -1
            normal[normal==-(height-1)] = 1
            normal[normal==-(width-1)] = 1
            dist = normal.abs().sum().sqrt()
            edge_features.append(torch.cat([normal, dist.unsqueeze(0)]))
            if dist > max_dist: max_dist = dist
        self.edge_features = torch.stack(edge_features).type(torch.float32).to(device=self.on_device)
        self.edge_features[:, -1] = self.edge_features[:, -1] / max_dist  # Normalize distances by longest distance

    def generate_edge_features(
        self,
        height: int = 64,
        width: int = 64
    ) -> Tensor:
        """
        Generate edge features.

        Returns a n x 3 array where row i contains (x_j - x_i) / |x_j - x_i|
        (node coordinates) and n is the number of nodes.
        Here, j and i are the node indices contained in row i of the edges1 and
        edges2 inputs. The second output is |x_j - x_i|.

        Arguments:
            points: n x 3 numpy array of point coordinates
            edges1: numpy array containing indices of source nodes for every edge
            edges2: numpy array containing indices of dest nodes for every edge
        Returns:
            n x 3 numpy array containing x_j - x_i
            n dimensional numpy array containing |x_j - x_i|

        """
        import numpy as np
        
        edges1, edges2 = self.graph.edges()
        edges1, edges2 = edges1.cpu().numpy(), edges2.cpu().numpy()
        
        points = []
        for node in self.graph.nodes():
            points.append([node//height, node%width])
        points = np.array(points)

        rel_position = []
        rel_position_norm = []
        nedges = len(edges1)
        for i in range(nedges):
            diff = points[edges2[i], :] - points[edges1[i], :]
            ndiff = np.linalg.norm(diff)
            rel_position.append(diff / ndiff)
            rel_position_norm.append(ndiff)
        return torch.tensor(np.array(rel_position, dtype=np.float32))#, rel_position_norm

    def update_nodes_and_edges(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
    ) -> Tensor:
        node_features = self.node_encoder(node_features)
        edge_features = self.edge_encoder(edge_features)
        for _ in range(self.message_passing_steps):
            node_features = self.processor(node_features, edge_features, graph)
        x = self.node_decoder(node_features)
        return x
    
    def _prepare_inputs(
        self,
        constants: torch.Tensor = None,
        prescribed: torch.Tensor = None,
        prognostic: torch.Tensor = None
    ) -> torch.Tensor:
        """
        return: Tensor of shape [B, (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(constants[:, 0])
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b t c h w -> b (t c) h w"))
        if prognostic is not None: tensors.append(einops.rearrange(prognostic, "b t c h w -> b (t c) h w"))
        return torch.cat(tensors, dim=1)

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
        b, t, d, h, w = prognostic.shape
        if b != self.batched_graph.batch_size: self.update_batched_graph(batch_size=b)
        outs = []

        for t in range(self.context_size, prognostic.shape[1]):
            
            t_start = max(0, t-(self.context_size))
            if t == self.context_size:
                # Initial condition
                prognostic_t = prognostic[:, t_start:t]
                x_t = self._prepare_inputs(
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
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )
        
            node_features = einops.rearrange(x_t, "b d h w -> (b h w) d")
            out = self.update_nodes_and_edges(
                node_features=node_features,
                edge_features=self.batched_edge_features,
                graph=self.batched_graph
            )
            out = prognostic_t[:, -1] + einops.rearrange(out, "(b h w) d -> b d h w", b=b, h=h, w=w, d=d)
            outs.append(out.cpu())
               
        return torch.stack(outs, dim=1)


class MeshGraphNetProcessor(nn.Module):
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
        activation_fn: nn.Module = nn.ReLU(),
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

        self.processor_layers = nn.ModuleList(layers)
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