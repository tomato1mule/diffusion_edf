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
from diffusion_edf.block import EquiformerBlock
from diffusion_edf.utils import multiply_irreps
from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge
from diffusion_edf.graph_parser import RadiusBipartite


class TensorField(torch.nn.Module):  
    cutoff: float

    @beartype
    def __init__(self,
        irreps_input: Union[str, o3.Irreps], 
        irreps_output: Union[str, o3.Irreps], 
        irreps_sh: Union[str, o3.Irreps], 
        num_heads: int, 
        fc_neurons: List[int],
        cutoff_radius: float,
        length_emb_dim: int,
        time_emb_dim: Optional[int],
        n_layers: int = 1,
        length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
        length_enc_kwargs: Dict =  {'n': 10000},
        max_neighbor: Optional[int] = None,
        irreps_mlp_mid: Union[str, o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0,
        infinite: bool = False):
        
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_heads: int = num_heads
        self.irreps_head: o3.Irreps = multiply_irreps(self.irreps_output, 1/self.num_heads, strict=True)
        self.fc_neurons: List[int] = fc_neurons
        self.infinite: bool = infinite
        self.cutoff_radius = cutoff_radius
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

        self.graph_parser = RadiusBipartite(r=self.cutoff_radius, 
                                            length_enc_dim=self.length_emb_dim,
                                            length_enc_type=length_enc_type, 
                                            length_enc_kwarg=length_enc_kwargs, 
                                            sh_irreps=self.irreps_sh, 
                                            max_neighbors=max_neighbor, 
                                            cutoff=True)
        

        #############################

        self.gnn = EquiformerBlock(irreps_src = self.irreps_input, 
                                   irreps_dst = self.irreps_output, 
                                   irreps_edge_attr = self.irreps_sh, 
                                   irreps_head = self.irreps_head,
                                   num_heads = self.num_heads, 
                                   fc_neurons = fc_neurons,
                                   irreps_mlp_mid = irreps_mlp_mid,
                                   attn_type = attn_type,
                                   alpha_drop = alpha_drop,
                                   proj_drop = proj_drop,
                                   drop_path_rate = drop_path_rate,
                                   src_bias = True,
                                   dst_bias = False, 
                                   dst_feature_layer = False,
                                   debug=True)

        self.register_buffer('zero_features', torch.zeros(1, self.emb_dim), persistent=False)
        

    def forward(self, query_points: FeaturedPoints,
                input_points_multiscale: List[FeaturedPoints],
                time_emb: Optional[List[torch.Tensor]] = None) -> FeaturedPoints:
        
        query_coord = query_points.x
        query_batch = query_points.b

        Nq, D = query_coord.shape
        assert query_batch.shape == (Nq, ) and D == 3
        if time_emb is not None:
            assert len(time_emb) == self.n_scales # time_emb[i]: Shape (nBatch, fc_neurons[0])
        Np, F = input_points_multiscale[-1].f.shape


        edge_srcs = torch.empty(0, device=query_coord.device, dtype=torch.long)
        edge_dsts = torch.empty(0, device=query_coord.device, dtype=torch.long)
        edge_vecs = torch.empty(0, 3, device=query_coord.device, dtype=query_coord.dtype)
        edge_scalars  = torch.empty(0, device=query_coord.device, dtype=query_coord.dtype)
        node_features = torch.empty(0, F, device=input_points_multiscale[-1].f.device, dtype=input_points_multiscale[-1].f.dtype)

        idx_begin: int = 0
        for n, (connect, radial) in enumerate(zip(self.pre_connect, self.pre_radial)):
            input_points_this_scale = input_points_multiscale[n]
            node_coord_this_scale = input_points_this_scale.x
            node_batch_this_scale = input_points_this_scale.b
            node_feature_this_scale = input_points_this_scale.f

            edge_src, edge_dst = connect(node_coord_src = node_coord_this_scale, 
                                         batch_src = node_batch_this_scale,
                                         node_coord_dst = query_coord,
                                         batch_dst = query_batch)
            edge_vec = node_coord_this_scale.index_select(0, edge_src) - query_coord.index_select(0, edge_dst)
            edge_length = edge_vec.norm(dim=1, p=2)
            in_range_idx = (edge_length > self.offsets[n]).nonzero().squeeze(-1)
            edge_src, edge_dst, edge_vec, edge_length = edge_src[in_range_idx], edge_dst[in_range_idx], edge_vec[in_range_idx], edge_length[in_range_idx]
            edge_src = edge_src + idx_begin
            idx_begin = idx_begin + len(node_coord_this_scale)
            edge_scalar = radial(edge_length)
            if time_emb is not None:
                time_emb_ = time_emb[n] # nBatch, nDim
                time_emb_ = time_emb_.index_select(0, query_batch) # Nq, nDim
                time_emb_ = time_emb_.index_select(0, edge_dst)    # nEdge, nDim
                edge_scalar = edge_scalar + time_emb_
            else:
                time_emb_ = None
            
            edge_srcs = torch.cat([edge_srcs, edge_src], dim=-1)
            edge_dsts = torch.cat([edge_dsts, edge_dst], dim=-1)
            edge_vecs = torch.cat([edge_vecs, edge_vec], dim=-2)
            edge_scalars = torch.cat([edge_scalars, edge_scalar], dim=-2)
            node_features = torch.cat([node_feature_this_scale], dim=-2)

        edge_attrs = self.spherical_harmonics(edge_vecs)

        node_feature_dst = torch.zeros(Nq, self.emb_dim, device=node_features.device, dtype=node_features.dtype)
        if len(edge_srcs) > 0:
            node_feature_dst = self.gnn(node_input_src = node_features,
                                        node_input_dst = node_feature_dst,
                                        batch_dst = query_batch,
                                        edge_src = edge_srcs,
                                        edge_dst = edge_dsts,
                                        edge_attr = edge_attrs,
                                        edge_scalars = edge_scalars)
        else:
            warnings.warn("No query point has neighborhood!")

        
        return None