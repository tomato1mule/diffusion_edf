from typing import List, Optional, Union, Tuple, Iterable, Callable
import math
import warnings
import numpy as np

import torch
from torch_cluster import fps, radius_graph
from torch_scatter import scatter_softmax
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from e3nn import o3
from e3nn.util.jit import compile_mode

from diffusion_edf.equiformer.layer_norm import EquivariantLayerNormV2
from diffusion_edf.equiformer.tensor_product_rescale import LinearRS
from diffusion_edf.equiformer.graph_attention_transformer import SeparableFCTP

from diffusion_edf.embedding import NodeEmbeddingNetwork
from diffusion_edf.block import EquiformerBlock, EdfExtractor
from diffusion_edf.connectivity import FpsPool, RadiusGraph, RadiusConnect
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff
from diffusion_edf.utils import multiply_irreps, ParityInversionSh
from diffusion_edf.skip import ProjectIfMismatch
from diffusion_edf.unet import EdfUnet, EDF
from diffusion_edf.query_model import QueryModel
from diffusion_edf.wigner import TransformFeatureQuaternion





#@compile_mode('script')
class ScoreModel(torch.nn.Module):
    def __init__(self, 
                 irreps_input: o3.Irreps,
                 irreps_emb_init: o3.Irreps,
                 irreps_sh: o3.Irreps,
                 fc_neurons_init: List[int],
                 num_heads: int,
                 n_scales: int,
                 pool_ratio: float,
                 dim_mult: List[Union[float, int]], 
                 n_layers: int,
                 gnn_radius: Union[float, List[float]],
                 cutoff_radius: Union[float, List[float]],
                 weight_feature_dim: int,
                 query_downsample_ratio: float,
                 device: Union[str, torch.device],
                 alpha_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 drop_path_rate: float = 0.0,
                 irreps_mlp_mid: int = 3,
                 deterministic: bool = False,
                 ):
        super().__init__()

        self.key_model = EDF(irreps_input=irreps_input,
                             irreps_emb_init=irreps_emb_init,
                             irreps_sh=irreps_sh,
                             fc_neurons_init=fc_neurons_init,
                             num_heads=num_heads,
                             n_scales=n_scales,
                             pool_ratio=pool_ratio,
                             dim_mult=dim_mult,
                             n_layers=n_layers,
                             gnn_radius=gnn_radius,
                             cutoff_radius=cutoff_radius,
                             alpha_drop=alpha_drop,
                             proj_drop=proj_drop,
                             drop_path_rate=drop_path_rate,
                             irreps_mlp_mid=irreps_mlp_mid,
                             deterministic=deterministic).to(device)

        self.query_model = QueryModel(irreps_input=irreps_input,
                                      irreps_emb_init=irreps_emb_init,
                                      irreps_sh=irreps_sh,
                                      fc_neurons_init=fc_neurons_init,
                                      num_heads=num_heads,
                                      n_scales=n_scales,
                                      pool_ratio=pool_ratio,
                                      dim_mult=dim_mult,
                                      n_layers=n_layers,
                                      gnn_radius=gnn_radius,
                                      cutoff_radius=cutoff_radius,
                                      alpha_drop=alpha_drop,
                                      proj_drop=proj_drop,
                                      drop_path_rate=drop_path_rate,
                                      irreps_mlp_mid=irreps_mlp_mid,
                                      weight_feature_dim=weight_feature_dim,
                                      query_downsample_ratio=query_downsample_ratio,
                                      deterministic=deterministic).to(device)
        
        irreps_score = []
        for mul, (l, p) in self.key_model.irreps_emb:
            if l == 1:
                assert p == 1
                irreps_score.append((mul, (l, p)))
        self.irreps_score = o3.Irreps(irreps_score) # Only spin-1 components
        self.n_irreps_score = self.irreps_score.dim // 3
        
        self.lin_vel_tp = SeparableFCTP(irreps_node_input = self.key_model.irreps_emb,
                                        irreps_edge_attr = self.query_model.irreps_emb, 
                                        irreps_node_output = o3.Irreps("1x0e") + self.irreps_score,  # Append 1x0e to avoid torch jit error. TODO: Remove this
                                        fc_neurons = None, 
                                        use_activation = True, 
                                        #norm_layer = 'layer', 
                                        norm_layer = None,
                                        internal_weights = True).to(device)
        #self.lin_vel_proj = LinearRS(irreps_in = self.irreps_score, irreps_out = o3.Irreps("1x1e"), bias=False, rescale=False).to(device)
        self.ang_vel_tp = SeparableFCTP(irreps_node_input = self.key_model.irreps_emb,
                                        irreps_edge_attr = self.query_model.irreps_emb, 
                                        irreps_node_output = o3.Irreps("1x0e") + self.irreps_score,  # Append 1x0e to avoid torch jit error. TODO: Remove this                     
                                        fc_neurons = None, 
                                        use_activation = True, 
                                        #norm_layer = 'layer', 
                                        norm_layer = None,
                                        internal_weights = True).to(device)
        #self.ang_vel_proj = LinearRS(irreps_in = self.irreps_score, irreps_out = o3.Irreps("1x1e"), bias=False, rescale=False).to(device)

        self.transform_irreps = TransformFeatureQuaternion(irreps = o3.Irreps(self.key_model.irreps_emb), device=device)

    @torch.jit.export
    def _get_query(self, node_feature: torch.Tensor, 
                   node_coord: torch.Tensor, 
                   batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:        
        query_weight, query_feature, query_coord, query_batch = self.query_model(node_feature=node_feature, node_coord=node_coord, batch=batch)

        return query_weight, query_feature, query_coord, query_batch # Shape: (N_query,), (N_query, D), (N_query, 3), (N_query,) 
    
    @torch.jit.export
    def _get_key(self, query_points: torch.Tensor,
                 query_batch: torch.Tensor,
                 gnn_features: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.key_model(query_points=query_points, query_batch=query_batch, gnn_features=gnn_features)
    
    @torch.jit.export
    def _get_gnn_features(self, node_feature: torch.Tensor, 
                          node_coord: torch.Tensor, 
                          batch: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.key_model.get_gnn_features(node_feature=node_feature, node_coord=node_coord, batch=batch)  # [(node_feature, node_coord, node_batch)]
    
    @torch.jit.export
    def get_score(self, T: torch.Tensor,
                  query: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                  key_gnn_features: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        query_weight, query_feature, query_coord, batch = query # Shape: (N_query,), (N_query, D), (N_query, 3), (N_query,) 
        assert query_weight.ndim + 1 == query_feature.ndim == query_coord.ndim == batch.ndim + 1 == 2 
        assert T.ndim == 2 and T.shape[-1] == 7

        N_T = len(T)
        N_Q = len(query_feature)
        N_D = query_feature.shape[-1]

        q, X = T[...,:4], T[...,4:] # (Nt,4), (Nt,3)
        query_coord_transformed: torch.Tensor = quaternion_apply(q.unsqueeze(-2), query_coord) # (Nt, 1, 4) x (Nq, 3) -> (Nt, Nq, 3)
        query_coord_transformed = query_coord_transformed + X.unsqueeze(-2) # (Nt, Nq, 3) + (Nt, 1, 3) -> (Nt, Nq, 3)
        query_feature_transformed = self.transform_irreps(query_feature, q).contiguous().view(N_T*N_Q, N_D) # (Nq, D) x (Nt, 4) -> (Nt * Nq, D)
        batch_repeat = batch.expand(N_T,N_Q).contiguous().view(-1) # (N_T*N_Q,)

        key_feature: torch.Tensor = self._get_key(query_points=query_coord_transformed.view(N_T * N_Q ,3), query_batch=batch_repeat, gnn_features=key_gnn_features)# (N_T * N_Q, N_D)

        lin_vel: torch.Tensor = self.lin_vel_tp(query_feature_transformed, key_feature, edge_scalars=None, batch=None) # batch does nothing unless you use batchnorm
        ang_spin: torch.Tensor = self.ang_vel_tp(query_feature_transformed, key_feature, edge_scalars=None, batch=None) # batch does nothing unless you use batchnorm
        
        lin_vel, ang_spin = lin_vel[..., 1:], ang_spin[..., 1:]                      # Discard the placeholder 1x0e feature to avoid torch jit error. TODO: Remove this

        lin_vel = lin_vel.view(N_T, N_Q, self.n_irreps_score, 3).mean(dim=-2)    # (N_T, N_Q, 3), Project multiple nx1e -> 1x1e 
        ang_spin = ang_spin.view(N_T, N_Q, self.n_irreps_score, 3).mean(dim=-2) # (N_T, N_Q, 3), Project multiple nx1e -> 1x1e 

        qinv: torch.Tensor = quaternion_invert(q.unsqueeze(-2)) # (N_T, 1, 4)
        qinv = qinv / torch.norm(qinv, dim=-1, keepdim=True)
        lin_vel = quaternion_apply(qinv, lin_vel) # (N_T, N_Q, 3)
        ang_spin = quaternion_apply(qinv, ang_spin) # (N_T, N_Q, 3)
        ang_orbital = torch.cross(query_coord.unsqueeze(0), lin_vel, dim=-1) # (N_T, N_Q, 3)

        lin_vel = torch.einsum('q,tqi->ti', query_weight, lin_vel) # (N_T, 3)
        ang_vel = torch.einsum('q,tqi->ti', query_weight, ang_orbital) + torch.einsum('q,tqi->ti', query_weight, ang_spin) # (N_T, 3)

        return lin_vel, ang_vel

    def forward(self, T: torch.Tensor,
                key_feature: torch.Tensor, key_coord: torch.Tensor, key_batch: torch.Tensor,
                query_feature: torch.Tensor, query_coord: torch.Tensor, query_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self._get_query(node_feature=query_feature, node_coord=query_coord, batch=query_batch)
        key_gnn_features = self._get_gnn_features(node_feature=key_feature, node_coord=key_coord, batch=key_batch)

        lin_vel, ang_vel = self.get_score(T=T, query=query, key_gnn_features=key_gnn_features)

        return lin_vel, ang_vel