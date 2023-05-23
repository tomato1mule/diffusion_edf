from typing import List, Optional, Union, Tuple, Dict
import warnings
import math
from beartype import beartype

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from einops import rearrange

from diffusion_edf.equiformer.drop import GraphDropPath, EquivariantDropout
from diffusion_edf.equiformer.tensor_product_rescale import FullyConnectedTensorProductRescale, LinearRS, FullyConnectedTensorProductRescaleSwishGate
from diffusion_edf.equiformer.layer_norm import EquivariantLayerNormV2
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first

from diffusion_edf.graph_attention import GraphAttentionMLP
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff
from diffusion_edf.gnn_block import EquiformerBlock
from diffusion_edf.utils import multiply_irreps
from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge
from diffusion_edf.graph_parser import RadiusBipartite


class TensorField(torch.nn.Module):  
    @beartype
    def __init__(self,
        irreps_input: Union[str, o3.Irreps], 
        irreps_output: Union[str, o3.Irreps], 
        irreps_sh: Union[str, o3.Irreps], 
        num_heads: int, 
        fc_neurons: List[int],
        length_emb_dim: int,
        time_emb_dim: Optional[int],
        r_maxcut: float,
        r_mincut_nonscalar: Optional[float],
        r_mincut_scalar: Optional[float] = None,
        n_layers: int = 1,
        length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
        length_enc_kwargs: Dict =  {}, # {'n': 10000},
        max_neighbor: Optional[int] = None,
        irreps_mlp_mid: Union[str, o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0):
        
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons
        self.r_maxcut = r_maxcut
        self.r_mincut_nonscalar = r_mincut_nonscalar
        self.r_mincut_scalar = r_mincut_scalar
        self.n_layers = n_layers
        assert self.n_layers >= 1
        if self.n_layers > 1:
            raise NotImplementedError

        self.length_emb_dim = length_emb_dim
        self.time_emb_dim = time_emb_dim
        assert self.length_emb_dim > 0, f"{self.length_emb_dim}"
        if self.time_emb_dim is None:
            assert fc_neurons[0] == self.length_emb_dim, f"{fc_neurons[0]}"
        else:
            assert self.time_emb_dim > 0, f"{self.time_emb_dim}"
            assert fc_neurons[0] == self.length_emb_dim + self.time_emb_dim, f"{fc_neurons[0]}"

        self.graph_parser = RadiusBipartite(r_maxcut=self.r_maxcut,
                                            r_mincut_nonscalar=self.r_mincut_nonscalar,
                                            r_mincut_scalar=self.r_mincut_scalar, 
                                            length_enc_dim=self.length_emb_dim,
                                            length_enc_type=length_enc_type, 
                                            length_enc_kwarg=length_enc_kwargs, 
                                            sh_irreps=self.irreps_sh, 
                                            max_neighbors=max_neighbor)

        self.gnn_block = EquiformerBlock(irreps_src = self.irreps_input, 
                                         irreps_dst = self.irreps_output, 
                                         irreps_emb = None, # equal to irreps_dst
                                         irreps_output = None, # equal to irreps_dst
                                         irreps_edge_attr = self.irreps_sh, 
                                         num_heads = self.num_heads, 
                                         fc_neurons = fc_neurons,
                                         irreps_mlp_mid = irreps_mlp_mid,
                                         attn_type = attn_type,
                                         alpha_drop = alpha_drop,
                                         proj_drop = proj_drop,
                                         drop_path_rate = drop_path_rate,
                                         use_dst_feature = False,
                                         skip_connection = True,
                                         bias = True,
                                         debug=True)
        
    def forward(self, query_points: FeaturedPoints,
                input_points: FeaturedPoints,
                time_emb: Optional[torch.Tensor] = None,
                max_neighbor: Optional[int] = None) -> FeaturedPoints:
        assert query_points.x.ndim == 2 # (Nq, 3)
        assert input_points.x.ndim == 2 # (Np, 3)
        assert time_emb.ndim == 2 # (Batch, tEmb)

        if max_neighbor is None:
            max_neighbor = len(input_points.x)

        graph_edge: GraphEdge = self.graph_parser(src=input_points, dst=query_points, max_neighbor=max_neighbor)
        if isinstance(time_emb, torch.Tensor):
            assert self.time_emb_dim is not None
            assert self.time_emb_dim == time_emb.shape[-1]
            time_emb = time_emb.index_select(0, query_points.b)            # (Nq, tEmb)
            time_emb = time_emb.index_select(0, graph_edge.edge_dst)       # (nEdge, tEmb)
            edge_scalars = torch.cat([graph_edge.edge_scalars, time_emb], dim=-1)  # (nEdge, tEmb + lEmb)
            graph_edge = GraphEdge(edge_src=graph_edge.edge_src, 
                                   edge_dst=graph_edge.edge_dst,
                                   edge_length=graph_edge.edge_length,
                                   edge_attr=graph_edge.edge_attr,
                                   edge_scalars=edge_scalars,
                                   edge_weight_scalar=graph_edge.edge_weight_scalar,
                                   edge_weight_nonscalar=graph_edge.edge_weight_nonscalar)

        output_points: FeaturedPoints = self.gnn_block(src_points=input_points,
                                                       dst_points=query_points,
                                                       graph_edge=graph_edge)
        
        return output_points
        