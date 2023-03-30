from typing import List, Optional, Union, Tuple
import math

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from diffusion_edf.block import EquiformerBlock, PoolingBlock, RadiusGraphBlock, sort_irreps_even_first
from diffusion_edf.connectivity import FpsPool, RadiusGraph, RadiusConnect
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff
from diffusion_edf.util import multiply_irreps

@compile_mode('script')
class EdfUnet(torch.nn.Module):  
    def __init__(self,
        irreps: List[o3.Irreps],
        irreps_edge_attr: List[o3.Irreps], 
        num_heads: List[int], 
        fc_neurons: List[List[int]],
        radius: List[float],
        pool_ratio: List[float],
        n_layers: List[int],
        deterministic: bool = False,
        pool_method: Union[Optional[str], List[Optional[str]]] = 'fps',
        irreps_mlp_mid: Union[Union[o3.Irreps, int], List[Union[o3.Irreps, int]]] = 3,
        attn_type: Union[str, List[str]] = 'mlp',
        alpha_drop: Union[float, List[float]] = 0.1,
        proj_drop: Union[float, List[float]] = 0.1,
        drop_path_rate: Union[float, List[float]] = 0.0):
        
        super().__init__()
        self.irreps: List[o3.Irreps] = [o3.Irreps(irrep) for irrep in irreps]
        self.irreps_edge_attr: List[o3.Irreps] = [o3.Irreps(irrep) for irrep in irreps_edge_attr]
        self.num_heads: List[int] = num_heads
        self.irreps_head: List[o3.Irreps] = [multiply_irreps(irrep, 1/self.num_heads) for irrep in self.irreps]
        self.fc_neurons: List[List[int]] = fc_neurons
        self.radius: List[float] = radius
        self.pool_ratio: List[float] = pool_ratio
        self.n_layers: List[int] = n_layers
        self.deterministic: bool = deterministic
        
        self.n_scales = len(self.irreps)
        if isinstance(pool_method, list):
            self.pool_method: List[Optional[str]] = pool_method
        else:
            self.pool_method: List[Optional[str]] = [pool_method for _ in range(self.n_scales)]
        if isinstance(irreps_mlp_mid, list):
            self.irreps_mlp_mid: List[Union[o3.Irreps, int]] = irreps_mlp_mid
        else:
            self.irreps_mlp_mid: List[Union[o3.Irreps, int]] = [irreps_mlp_mid for _ in range(self.n_scales)]
        if isinstance(attn_type, list):
            self.attn_type: List[str] = attn_type
        else:
            self.attn_type: List[str] = [attn_type for _ in range(self.n_scales)]
        if isinstance(alpha_drop, list):
            self.alpha_drop: List[float] = alpha_drop
        else:
            self.alpha_drop: List[float] = [alpha_drop for _ in range(self.n_scales)]
        if isinstance(proj_drop, list):
            self.proj_drop: List[float] = proj_drop
        else:
            self.proj_drop: List[float] = [proj_drop for _ in range(self.n_scales)]
        if isinstance(drop_path_rate, list):
            self.drop_path_rate: List[float] = drop_path_rate
        else:
            self.drop_path_rate: List[float] = [drop_path_rate for _ in range(self.n_scales)]

        assert self.n_scales == len(self.irreps) == len(self.irreps_edge_attr) == len(self.num_heads) == len(self.irreps_head) == len(self.fc_neurons) == len(self.radius) == len(self.pool_ratio) == len(self.n_layers) == len(self.pool_method) == len(self.irreps_mlp_mid) == len(self.attn_type) == len(self.alpha_drop) == len(self.proj_drop) == len(self.drop_path_rate)
        for n in range(self.n_scales):
            if self.pool_ratio[n] == 1.0:
                assert self.pool_method[n] is None
            else:
                assert self.pool_method[n] is not None
            assert self.n_layers[n] >= 1

        self.down_layers = torch.nn.ModuleList()
        self.up_layers = torch.nn.ModuleList()
        self.mid_layers = torch.nn.ModuleList()
        
        for s in range(self.n_scales):
            layers = torch.nn.ModuleList()

            if self.pool_method[n] is not None:
                layers.append(
                    PoolingBlock(irreps_src = self.irreps[n],
                                 irreps_dst = self.irreps[n],
                                 irreps_edge_attr = self.irreps_edge_attr[n],
                                 irreps_head = self.irreps_head[n],
                                 num_heads = self.num_heads[n],
                                 fc_neurons = self.fc_neurons[n],
                                 pool_radius = radius[n],
                                 pool_ratio = self.pool_ratio[n],
                                 pool_method = self.pool_method[n],
                                 deterministic = self.deterministic,
                                 irreps_mlp_mid = irreps_mlp_mid[n],
                                 attn_type = attn_type[n],
                                 alpha_drop = alpha_drop[n],
                                 proj_drop = proj_drop[n],
                                 drop_path_rate = drop_path_rate[n])
                )
            
            n_self_connecting_layers = self.n_layers[n] - (self.pool_method[n] is not None)
            if n_self_connecting_layers >= 1:
                layers.append(
                    RadiusGraphBlock(irreps = self.irreps[n],
                                     irreps_edge_attr = self.irreps_edge_attr[n],
                                     irreps_head = self.irreps_head[n],
                                     num_heads = self.num_heads[n],
                                     fc_neurons = self.fc_neurons[n],
                                     radius = radius[n],
                                     n_layers = n_self_connecting_layers[n],
                                     irreps_mlp_mid = irreps_mlp_mid[n],
                                     attn_type = attn_type[n],
                                     alpha_drop = alpha_drop[n],
                                     proj_drop = proj_drop[n],
                                     drop_path_rate = drop_path_rate[n])
                )



    def forward(self, node_feature: torch.Tensor,
                node_coord: torch.Tensor,
                batch: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            output = layer(node_feature = node_feature, node_coord = node_coord, batch = batch)
            outputs.append(output)
            node_feature, node_coord, batch = output[0], output[1], output[7]
        
        return outputs