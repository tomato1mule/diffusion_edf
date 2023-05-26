from typing import List, Optional, Union, Tuple, Dict
import warnings
import math
from beartype import beartype

import torch
from e3nn import o3


from diffusion_edf.gnn_block import EquiformerBlock
from diffusion_edf.utils import multiply_irreps
from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge, set_graph_edge_attribute
from diffusion_edf.graph_parser import RadiusBipartite


class MultiscaleTensorField(torch.nn.Module):
    r_cluster_multiscale: List[float]
    n_scales: int

    @beartype
    def __init__(self,
        irreps_input: Union[str, o3.Irreps], 
        irreps_output: Union[str, o3.Irreps], 
        irreps_sh: Union[str, o3.Irreps], 
        num_heads: int, 
        fc_neurons: List[int],
        length_emb_dim: int,
        context_emb_dim: Optional[int],                        # Explicitly set to None if not want to use context embedding
        irreps_query: Optional[Union[str, o3.Irreps]],      # Explicitly set to None if not want to use query features
        r_cluster_multiscale: List[float],
        contexted_edge_scalars: bool = True,
        n_scales: Optional[int] = None,                     # This parameter is just for double checking. If set to None, it will be automatically set to r_cluster_multiscale.
        n_layers: int = 1,
        irreps_mlp_mid: Union[str, o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0):

        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_sh = o3.Irreps(irreps_sh)
        if irreps_query is not None:
            self.use_dst_feature = True
            self.irreps_query = o3.Irreps(irreps_query)
        else:
            self.use_dst_feature = False
            self.irreps_query = None
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons
        

        self.length_emb_dim = length_emb_dim
        self.context_emb_dim = context_emb_dim
        self.contexted_edge_scalars = contexted_edge_scalars
        assert self.length_emb_dim > 0, f"{self.length_emb_dim}"

        if self.fc_neurons[0] == -1:
            if self.context_emb_dim is None:
                self.fc_neurons[0] = self.length_emb_dim
            else:
                self.fc_neurons[0] = self.length_emb_dim + self.context_emb_dim

        if self.context_emb_dim is None:
            assert fc_neurons[0] == self.length_emb_dim, f"{fc_neurons[0]} != {self.length_emb_dim}"
        else:
            assert self.context_emb_dim > 0, f"{self.context_emb_dim}"
            assert fc_neurons[0] == self.length_emb_dim + self.context_emb_dim, f"{fc_neurons[0]} != {self.length_emb_dim} + {self.context_emb_dim}"


        self.r_cluster_multiscale = r_cluster_multiscale
        if n_scales is None:
            self.n_scales = len(self.r_cluster_multiscale)
        else:
            assert n_scales == len(self.r_cluster_multiscale)
            self.n_scales = n_scales

        # !!!!!!!!!!!!!!TODO!!!!!!!!!!!! 
        # min_offset = self.r_cluster_multiscale[0] * 0.01
        # self.mincut_offsets = [min_offset] + [max(min_offset, offset - 0.2*(cutoff - offset)) for offset, cutoff in zip(self.r_cluster_multiscale[:-1], self.r_cluster_multiscale[1:])],

        self.graph_parsers = torch.nn.ModuleList()
        for n in range(self.n_scales):
            self.graph_parsers.append(
                RadiusBipartite(
                    # r_cutoff=[self.mincut_offsets[n], ??, ?? , self.r_cluster_multiscale[n]],
                    r_cutoff=self.r_cluster_multiscale[n],
                    irreps_sh=self.irreps_sh,
                    length_enc_dim=self.length_emb_dim,
                    length_enc_type='BesselBasisEncoder',
                )
            )
        
        assert self.n_layers >= 1
        self.n_layers = n_layers
        self.gnn_block_init = EquiformerBlock(irreps_src = self.irreps_input, 
                                              irreps_dst = self.irreps_query, 
                                              irreps_emb = self.irreps_input, 
                                              irreps_output = self.irreps_output if self.n_layers == 1 else self.irreps_input, 
                                              irreps_edge_attr = self.irreps_sh, 
                                              num_heads = self.num_heads, 
                                              fc_neurons = fc_neurons,
                                              irreps_mlp_mid = irreps_mlp_mid,
                                              attn_type = attn_type,
                                              alpha_drop = alpha_drop,
                                              proj_drop = proj_drop,
                                              drop_path_rate = drop_path_rate,
                                              use_dst_feature = self.use_dst_feature,
                                              skip_connection = True,
                                              bias = True,
                                              use_src_w = True,
                                              use_dst_w = False)
        
        self.gnn_blocks = torch.nn.ModuleList()
        for n in range(self.n_layers-1):
            self.gnn_blocks.append(
                EquiformerBlock(irreps_src = self.irreps_input, 
                                irreps_dst = self.irreps_input, 
                                irreps_emb = self.irreps_input, 
                                irreps_output = self.irreps_output if n == self.n_layers-1 else self.irreps_input, 
                                irreps_edge_attr = self.irreps_sh, 
                                num_heads = self.num_heads, 
                                fc_neurons = fc_neurons,
                                irreps_mlp_mid = irreps_mlp_mid,
                                attn_type = attn_type,
                                alpha_drop = alpha_drop,
                                proj_drop = proj_drop,
                                drop_path_rate = drop_path_rate,
                                use_dst_feature = True,
                                skip_connection = True,
                                bias = True,
                                use_src_w = True,
                                use_dst_w = False)
            )
        
    def forward(self, query_points: FeaturedPoints,
                input_points_multiscale: List[FeaturedPoints],
                context_emb: Optional[torch.Tensor] = None,
                max_neighbors: Optional[int] = 1000) -> FeaturedPoints:
        assert query_points.x.ndim == 2 # (Nq, 3)
        if context_emb is not None:
            assert context_emb.ndim == 2 # (Nq, tEmb)
        if self.context_emb_dim is not None:
            assert isinstance(context_emb, torch.Tensor)
            assert self.context_emb_dim == context_emb.shape[-1], f"{self.context_emb_dim} != {context_emb.shape[-1]} of {context_emb.shape}"
        device: torch.device = query_points.x.device

        edge_srcs = torch.empty(0, device=device, dtype=torch.long)
        edge_dsts = torch.empty(0, device=device, dtype=torch.long)
        for n, graph_parser in enumerate(self.graph_parsers):
            input_points: FeaturedPoints = input_points_multiscale[n]
            graph_edge: GraphEdge = graph_parser(src=input_points, dst=query_points, max_neighbors=max_neighbors)
            if self.contexted_edge_scalars is True:
                context_emb_ = context_emb.index_select(0, graph_edge.edge_dst)       # (nEdge, cEmb)
                edge_scalars = torch.cat([graph_edge.edge_scalars, context_emb_], dim=-1)  # (nEdge, tEmb + cEmb)
                graph_edge = set_graph_edge_attribute(graph_edge, edge_scalars=edge_scalars)
            
            TODO
                

        output_points: FeaturedPoints = self.gnn_block_init(src_points=input_points,
                                                            dst_points=query_points,
                                                            graph_edge=graph_edge)
        for block in self.gnn_blocks:
            output_points: FeaturedPoints = block(src_points=input_points,
                                                  dst_points=output_points,
                                                  graph_edge=graph_edge)
        
        return output_points
        