from typing import List, Optional, Union, Tuple
import warnings
import math

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from einops import rearrange

from diffusion_edf.equiformer.drop import GraphDropPath, EquivariantDropout
from diffusion_edf.equiformer.tensor_product_rescale import FullyConnectedTensorProductRescale, LinearRS, FullyConnectedTensorProductRescaleSwishGate
from diffusion_edf.equiformer.layer_norm import EquivariantLayerNormV2
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first

from diffusion_edf.graph_attention import GraphAttentionMLP
from diffusion_edf.connectivity import FpsPool, RadiusGraph, RadiusConnect
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff
from diffusion_edf.block import EquiformerBlock
from diffusion_edf.utils import SinusoidalPositionEmbeddings


#@compile_mode('script')
# class EdfExtractor(torch.nn.Module):  
#     def __init__(self,
#         irreps_inputs: List[o3.Irreps], 
#         fc_neurons_inputs: List[List[int]],
#         irreps_emb: o3.Irreps,
#         irreps_edge_attr: o3.Irreps, 
#         irreps_head: o3.Irreps,
#         num_heads: int, 
#         fc_neurons: List[int],
#         n_layers: int,
#         cutoffs: List[float],
#         offsets: List[float],
#         query_radius: Optional[float] = None,
#         irreps_mlp_mid: Union[o3.Irreps, int] = 3,
#         attn_type: str = 'mlp',
#         alpha_drop: float = 0.1,
#         proj_drop: float = 0.1,
#         drop_path_rate: float = 0.0):
        
#         super().__init__()
#         self.irreps_inputs: List[o3.Irreps] = [o3.Irreps(irreps) for irreps in irreps_inputs]
#         self.fc_neurons_inputs: List[List[int]] = fc_neurons_inputs
#         self.n_scales: int = len(self.irreps_inputs)
#         self.cutoffs: List[float] = cutoffs
#         self.offsets: List[float] = offsets
#         assert len(self.offsets) == len(self.cutoffs) == len(self.fc_neurons_inputs) == self.n_scales

#         self.irreps_emb: o3.Irreps = o3.Irreps(irreps_emb)
#         self.emb_dim: int = self.irreps_emb.dim
#         self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
#         self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
#         self.num_heads: int = num_heads
#         self.fc_neurons: List[int] = fc_neurons
#         self.n_layers: int = n_layers
#         self.query_radius: Optional[float] = query_radius
#         assert self.n_layers >= 1

#         self.pre_connect = torch.nn.ModuleList()
#         self.pre_radial = torch.nn.ModuleList()
#         self.pre_layers = torch.nn.ModuleList()
#         for n in range(self.n_scales):
#             self.pre_connect.append(
#                 RadiusConnect(r=self.cutoffs[n], offset=None, max_num_neighbors= 1000) # TODO: offset=None -> self.offsets[n]
#             )
#             fc = self.fc_neurons_inputs[n]
#             self.pre_radial.append(
#                 GaussianRadialBasisLayerFiniteCutoff(num_basis=fc[0], 
#                                                      cutoff=self.cutoffs[n], 
#                                                      offset=self.offsets[n],
#                                                      soft_cutoff=True)
#             )
#             self.pre_layers.append(
#                 EquiformerBlock(irreps_src = self.irreps_inputs[n], 
#                                 irreps_dst = self.irreps_emb, 
#                                 irreps_edge_attr = self.irreps_edge_attr, 
#                                 irreps_head = self.irreps_head,
#                                 num_heads = self.num_heads, 
#                                 fc_neurons = fc,
#                                 irreps_mlp_mid = irreps_mlp_mid,
#                                 attn_type = attn_type,
#                                 alpha_drop = alpha_drop,
#                                 proj_drop = proj_drop,
#                                 drop_path_rate = drop_path_rate,
#                                 src_bias = False,
#                                 dst_bias = True)
#             )

#         self.spherical_harmonics = o3.SphericalHarmonics(irreps_out = self.irreps_edge_attr, normalize = True, normalization='component')    
#         self.register_buffer('zero_features', torch.zeros(1, self.emb_dim), persistent=False)
#         self.proj = LinearRS(irreps_in = self.irreps_emb,
#                              irreps_out = self.irreps_emb,
#                              bias = True)


#     def forward(self, query_coord: torch.Tensor,
#                 query_batch_n_scale: torch.Tensor,
#                 node_feature: torch.Tensor,
#                 node_coord: torch.Tensor,
#                 node_batch_n_scale: torch.Tensor) -> torch.Tensor:
#         Ns, Nq, _ = query_coord.shape
#         node_feature_dst = torch.zeros(Nq, self.emb_dim, device=node_feature.device, dtype=node_feature.dtype)

#         for n, (connect, radial, layers) in enumerate(zip(self.pre_connect, self.pre_radial, self.pre_layers)):
#             edge_src, edge_dst = connect(node_coord_src = node_coord, 
#                                          batch_src = node_batch_n_scale,
#                                          node_coord_dst = query_coord[n],
#                                          batch_dst = query_batch_n_scale[n])
#             edge_vec = node_coord.index_select(0, edge_src) - query_coord[n].index_select(0, edge_dst)
#             edge_attr = self.spherical_harmonics(edge_vec)
#             edge_length = edge_vec.norm(dim=1, p=2)
#             edge_scalar = radial(edge_length)

#             node_feature_dst = node_feature_dst \
#                                + layers(node_input_src = node_feature,
#                                         node_input_dst = torch.zeros(Nq, self.emb_dim, device=node_feature.device, dtype=node_feature.dtype),
#                                         batch_dst = query_batch_n_scale[n],
#                                         edge_src = edge_src,
#                                         edge_dst = edge_dst,
#                                         edge_attr = edge_attr,
#                                         edge_scalars = edge_scalar)
        
#         return self.proj(node_feature_dst)













class EdfExtractorLight(torch.nn.Module):  
    def __init__(self,
        irreps_inputs: List[o3.Irreps], 
        irreps_emb: o3.Irreps,
        irreps_edge_attr: o3.Irreps, 
        irreps_head: o3.Irreps,
        num_heads: int, 
        fc_neurons: List[int],
        n_layers: int,
        cutoffs: List[float],
        offsets: List[float],
        query_radius: Optional[float] = None,
        irreps_mlp_mid: Union[o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0):
        
        super().__init__()
        self.irreps_inputs: List[o3.Irreps] = [o3.Irreps(irreps) for irreps in irreps_inputs]
        self.n_scales: int = len(self.irreps_inputs)
        self.cutoffs: List[float] = cutoffs
        self.offsets: List[float] = offsets
        assert len(self.offsets) == len(self.cutoffs) == self.n_scales

        self.irreps_emb: o3.Irreps = o3.Irreps(irreps_emb)
        self.emb_dim: int = self.irreps_emb.dim
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons
        self.n_layers: int = n_layers
        self.query_radius: Optional[float] = query_radius
        assert self.n_layers >= 1

        self.pre_connect = torch.nn.ModuleList()
        self.pre_radial = torch.nn.ModuleList()
        self.pre_layers = torch.nn.ModuleList()
        self.num_basis: List[int] = []
        for n in range(self.n_scales):
            self.pre_connect.append(
                RadiusConnect(r=self.cutoffs[n], offset=None, max_num_neighbors= 1000) # TODO: offset=None -> self.offsets[n]
            )
            self.pre_radial.append(
                torch.nn.Sequential(
                    GaussianRadialBasisLayerFiniteCutoff(num_basis=fc_neurons[0], 
                                                        cutoff=self.cutoffs[n], 
                                                        offset=self.offsets[n],
                                                        soft_cutoff=True),
                    torch.nn.Linear(fc_neurons[0], fc_neurons[0])
                )
            )
            self.num_basis.append(fc_neurons[0])
        self.gnn = EquiformerBlock(irreps_src = self.irreps_inputs[n], 
                                   irreps_dst = self.irreps_emb, 
                                   irreps_edge_attr = self.irreps_edge_attr, 
                                   irreps_head = self.irreps_head,
                                   num_heads = self.num_heads, 
                                   fc_neurons = fc_neurons,
                                   irreps_mlp_mid = irreps_mlp_mid,
                                   attn_type = attn_type,
                                   alpha_drop = alpha_drop,
                                   proj_drop = proj_drop,
                                   drop_path_rate = drop_path_rate,
                                   src_bias = False,
                                   dst_bias = True, debug=True)

        self.spherical_harmonics = o3.SphericalHarmonics(irreps_out = self.irreps_edge_attr, normalize = True, normalization='component')
        self.register_buffer('zero_features', torch.zeros(1, self.emb_dim), persistent=False)
        

    def forward(self, query_coord: torch.Tensor,
                query_batch: torch.Tensor,
                node_feature: torch.Tensor,
                node_coord: torch.Tensor,
                node_batch: torch.Tensor,
                node_scale_slice: List[int],
                time_emb: Optional[List[torch.Tensor]] = None,) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        Nq, D = query_coord.shape
        assert query_batch.shape == (Nq, ) and D == 3
        if time_emb is not None:
            assert len(time_emb) == self.n_scales # time_emb[i]: Shape (nBatch, fc_neurons[0])

        edge_srcs = torch.empty(0, device=query_coord.device, dtype=torch.long)
        edge_dsts = torch.empty(0, device=query_coord.device, dtype=torch.long)
        edge_vecs = torch.empty(0, 3, device=query_coord.device, dtype=query_coord.dtype)
        edge_scalars  = torch.empty(0, device=query_coord.device, dtype=query_coord.dtype)
        for n, (connect, radial) in enumerate(zip(self.pre_connect, self.pre_radial)):
            slice_start: int = node_scale_slice[n]
            slice_length: int = node_scale_slice[n+1] - slice_start
            node_coord_this_scale = torch.narrow(node_coord, dim=-2, start=slice_start, length=slice_length)
            node_batch_this_scale = torch.narrow(node_batch, dim=-1, start=slice_start, length=slice_length)

            edge_src, edge_dst = connect(node_coord_src = node_coord_this_scale, 
                                         batch_src = node_batch_this_scale,
                                         node_coord_dst = query_coord,
                                         batch_dst = query_batch)
            edge_vec = node_coord_this_scale.index_select(0, edge_src) - query_coord.index_select(0, edge_dst)
            edge_length = edge_vec.norm(dim=1, p=2)
            in_range_idx = (edge_length > self.offsets[n]).nonzero().squeeze(-1)
            edge_src, edge_dst, edge_vec, edge_length = edge_src[in_range_idx], edge_dst[in_range_idx], edge_vec[in_range_idx], edge_length[in_range_idx]
            edge_src = edge_src + slice_start
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

        edge_attrs = self.spherical_harmonics(edge_vecs)

        node_feature_dst = torch.zeros(Nq, self.emb_dim, device=node_feature.device, dtype=node_feature.dtype)
        if len(edge_srcs) > 0:
            node_feature_dst = self.gnn(node_input_src = node_feature,
                                        node_input_dst = node_feature_dst,
                                        batch_dst = query_batch,
                                        edge_src = edge_srcs,
                                        edge_dst = edge_dsts,
                                        edge_attr = edge_attrs,
                                        edge_scalars = edge_scalars)
        else:
            warnings.warn("No query point has neighborhood!")
        
        info = (edge_srcs.detach(), edge_dsts.detach())
        
        return node_feature_dst, info  # Shape: (Nq, D),  ((N_edges,), (N_edges,))