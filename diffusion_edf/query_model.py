from typing import List, Optional, Union, Tuple, Iterable, Callable
import math
import warnings
import numpy as np

import torch
from torch_cluster import fps, radius_graph
from torch_scatter import scatter_softmax
from e3nn import o3
from e3nn.util.jit import compile_mode

from diffusion_edf.equiformer.layer_norm import EquivariantLayerNormV2
from diffusion_edf.equiformer.tensor_product_rescale import LinearRS

from diffusion_edf.embedding import NodeEmbeddingNetwork
from diffusion_edf.extractor import EdfExtractorLight
from diffusion_edf.connectivity import FpsPool, RadiusGraph, RadiusConnect
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff
from diffusion_edf.utils import multiply_irreps, ParityInversionSh
from diffusion_edf.skip import ProjectIfMismatch
from diffusion_edf.unet import EDF, EdfUnet





#@compile_mode('script')
class QueryModel(EDF):
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
                 alpha_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 drop_path_rate: float = 0.0,
                 irreps_mlp_mid: int = 3,
                 deterministic: bool = False,
                 compile_head: bool = False,
                 attn_type: str = 'mlp',
                 ):
        super().__init__(irreps_input=irreps_input,
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
                         deterministic=deterministic,
                         compile_head=compile_head,
                         attn_type=attn_type
                         )
        self.query_downsample_ratio = query_downsample_ratio

        self.weight_field = EdfExtractorLight(
            irreps_inputs = self.gnn.irreps,
            irreps_emb = self.gnn.irreps[-1],
            irreps_edge_attr = self.gnn.irreps_edge_attr[-1],
            irreps_head = self.gnn.irreps_head[-1],
            num_heads = self.gnn.num_heads[-1],
            fc_neurons = self.gnn.fc_neurons[-1],
            n_layers = 1,
            cutoffs = self.cutoff_radius,
            offsets = [self.min_offset] + [max(self.min_offset, offset - 0.2*(cutoff - offset)) for offset, cutoff in zip(self.cutoff_radius[:-1], self.cutoff_radius[1:])],
            irreps_mlp_mid = irreps_mlp_mid,
            attn_type=attn_type,
            alpha_drop=alpha_drop, 
            proj_drop=proj_drop,
            drop_path_rate=drop_path_rate
        )
        self.pre_weight_irreps = o3.Irreps(f"{weight_feature_dim}x0e")
        self.weight_linear1 = LinearRS(irreps_in = self.gnn.irreps[-1], irreps_out = self.pre_weight_irreps, bias=True, rescale=True)
        self.weight_layernorm = EquivariantLayerNormV2(irreps = self.pre_weight_irreps)
        self.weight_linear2 = LinearRS(irreps_in = self.pre_weight_irreps, irreps_out = o3.Irreps("1x0e"), bias=True, rescale=True)        

    def _extract_weight_logits(self, query_coord: torch.Tensor,
                               query_batch_n_scale: torch.Tensor,
                               node_feature: torch.Tensor,
                               node_coord: torch.Tensor,
                               node_batch_n_scale: torch.Tensor) -> torch.Tensor:
        
        field_val, (edge_src, edge_dst) = self.weight_field(query_coord = query_coord, 
                                                            query_batch_n_scale = query_batch_n_scale,
                                                            node_feature = node_feature,
                                                            node_coord = node_coord,
                                                            node_batch_n_scale = node_batch_n_scale)
        field_val = self.weight_linear1(field_val)
        field_val = self.weight_layernorm(field_val)
        field_val = self.weight_linear2(field_val)
        
        return field_val.squeeze(-1)
    
    def _get_init_query_pos(self, node_coord: torch.Tensor, node_batch_n_scale: torch.Tensor, only_from_top_scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if only_from_top_scale:
            top_scale_node_idx = (node_batch_n_scale % self.n_scales == (self.n_scales-1)).nonzero().squeeze(-1)
            node_coord = torch.index_select(node_coord, dim=-2, index = top_scale_node_idx)
            batch = torch.index_select(node_batch_n_scale, dim=-1, index = top_scale_node_idx) // self.n_scales
        else:
            batch = node_batch_n_scale // self.n_scales

        node_dst_idx = fps(src=node_coord, batch=batch, ratio=self.query_downsample_ratio, random_start=not self.deterministic)
        query_coord = node_coord.index_select(index=node_dst_idx, dim=0)
        query_batch = batch.index_select(index=node_dst_idx, dim=0)

        return query_coord, query_batch
    
    # def _rbf(self, x1: torch.Tensor, x2: torch.Tensor, h: float) -> torch.Tensor:
    #     return torch.exp(-(x1-x2).square().sum(dim=-1)/h)

    # def _rbf_grad_x1(self, x1: torch.Tensor, x2: torch.Tensor, h: float) -> torch.Tensor:
    #     return -2/h * (x1-x2) * self.rbf(x1,x2,h).unsqueeze(-1)

    # def stein_vgd(self, x: torch.Tensor, batch: torch.Tensor, log_P: Callable, iters: int, lr: float):
    #     requires_grad = x.requires_grad
    #     if iters > 0:

    #         x1 = x.detach()
    #         graph = radius_graph(x1, r = torch.inf)
    #         if torch.are_deterministic_algorithms_enabled():
    #             med = (x1[graph[1]] - x1[graph[0]]).norm(dim=-1).cpu().median(dim=-1).values.to(x1.device)
    #         else:
    #             med = (x1[graph[1]] - x1[graph[0]]).norm(dim=-1).median(dim=-1).values

    #         h = med.square() / np.log(max(len(x1), 1))

    #         for i in range(iters):
    #             rkhs = self.rbf(x.unsqueeze(1), x.unsqueeze(0), h) # (Nq, Nq)
    #             rkhs_grad = self.rbf_grad_x1(x.unsqueeze(1), x.unsqueeze(0), h) # (Nq, Nq, 3)
    #             if not requires_grad:
    #                 x_ = x.detach().requires_grad_(True)
    #                 grad = torch.autograd.grad(log_P(x_).sum(dim=-1), x_, create_graph = False)[0] # (Nq, 3)
    #             else:
    #                 grad = torch.autograd.grad(log_P(x).sum(dim=-1), x, create_graph = True)[0] # (Nq, 3)
    #             phi = ((grad.unsqueeze(1) * rkhs.unsqueeze(-1)) + rkhs_grad).mean(dim=0)
    #             x = x + lr*phi

    #     return x
    
    def forward(self, node_feature: torch.Tensor, 
                node_coord: torch.Tensor, 
                batch: torch.Tensor,
                detach_info: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                                   Tuple[torch.Tensor, torch.Tensor], 
                                                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        gnn_outputs = self.get_gnn_outputs(node_feature=node_feature, node_coord=node_coord, batch=batch)
        node_feature, node_coord, node_batch_n_scale, edge_src, edge_dst = gnn_outputs

        query_coord, query_batch = self._get_init_query_pos(node_coord=node_coord, node_batch_n_scale=node_batch_n_scale, only_from_top_scale=True)
        query_coord_expanded = query_coord.expand(self.n_scales, len(query_coord), 3) # (Ns, Nq, 3)
        query_scale = torch.arange(self.n_scales, dtype=query_batch.dtype, device=query_batch.device)
        query_batch_n_scale = query_batch * query_scale.unsqueeze(-1) + query_scale.unsqueeze(-1) # (Ns, Nq)

        query_weight = self._extract_weight_logits(query_coord=query_coord_expanded, query_batch_n_scale=query_batch_n_scale,
                                                   node_feature=node_feature, node_coord=node_coord, node_batch_n_scale=node_batch_n_scale)
        print(query_weight.shape, query_coord.shape, query_batch.shape)
        query_weight = scatter_softmax(src = query_weight, index=query_batch)
        query_feature, query_info = self.extractor(query_coord = query_coord_expanded, 
                                                   query_batch_n_scale = query_batch_n_scale,
                                                   node_feature = node_feature,
                                                   node_coord = node_coord,
                                                   node_batch_n_scale = node_batch_n_scale)
        
        (edge_src_query, edge_dst_query) = query_info
        query = (query_weight, query_feature, query_coord_expanded, query_batch_n_scale)
        if detach_info:
            gnn_outputs = (node_feature.detach(), 
                           node_coord.detach(), 
                           node_batch_n_scale.detach(), 
                           edge_src.detach(), 
                           edge_dst.detach())
            query_info = (edge_src_query.detach(), edge_dst_query.detach())
        

        return query, query_info, gnn_outputs