from typing import List, Optional, Union, Tuple

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
import torch_geometric
from torch_scatter import scatter

from diffusion_edf.equiformer.drop import GraphDropPath, EquivariantDropout
from diffusion_edf.equiformer.tensor_product_rescale import FullyConnectedTensorProductRescale, LinearRS, FullyConnectedTensorProductRescaleSwishGate
from diffusion_edf.equiformer.layer_norm import EquivariantLayerNormV2
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first

from diffusion_edf.graph_attention import GraphAttentionMLP
from diffusion_edf.connectivity import FpsPool, RadiusGraph
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff


@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
        irreps_node_input: o3.Irreps,
        irreps_node_output: o3.Irreps, 
        irreps_mlp_mid: Optional[o3.Irreps] = None,
        proj_drop: float = 0.1, bias: bool = True, rescale: bool = True):
        
        super().__init__()
        self.irreps_node_input: o3.Irreps = o3.Irreps(irreps_node_input)
        self.irreps_mlp_mid: o3.Irreps = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output: o3.Irreps = o3.Irreps(irreps_node_output)
        self.irreps_node_attr = o3.Irreps("1x0e")
        
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=bias, rescale=rescale)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=bias, rescale=rescale)
        
        if not proj_drop:
            self.proj_drop = None
        else:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, drop_prob=proj_drop)
            
        
    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        node_attr = torch.ones_like(node_input[:, 0:1])
        node_output: torch.Tensor = self.fctp_1(node_input, node_attr)
        node_output: torch.Tensor = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output: torch.Tensor = self.proj_drop(node_output)
        return node_output






@compile_mode('script')
class EquiformerBlock(torch.nn.Module):  
    def __init__(self,
        irreps_src: o3.Irreps, 
        irreps_dst: o3.Irreps, 
        irreps_edge_attr: o3.Irreps, 
        irreps_head: o3.Irreps,
        num_heads: int, 
        fc_neurons: List[int],
        irreps_mlp_mid: Union[o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0,
        src_bias: bool = False,
        dst_bias: bool = True):
        
        super().__init__()
        self.irreps_src: o3.Irreps = o3.Irreps(irreps_src)
        self.irreps_dst: o3.Irreps = o3.Irreps(irreps_dst)
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons

        self.irreps_emb: o3.Irreps = self.irreps_dst
        assert num_heads*self.irreps_head.dim == self.irreps_emb.dim
        if isinstance(irreps_mlp_mid, o3.Irreps):
            self.irreps_mlp_mid: o3.Irreps = o3.Irreps(irreps_mlp_mid)
        elif isinstance(irreps_mlp_mid, int):
            self.irreps_mlp_mid = sort_irreps_even_first((self.irreps_emb * irreps_mlp_mid))[0].simplify()

        self.norm_1_src = EquivariantLayerNormV2(self.irreps_src)
        self.norm_1_dst = EquivariantLayerNormV2(self.irreps_dst)

        self.linear_src = LinearRS(self.irreps_src, self.irreps_emb, bias=src_bias)
        self.linear_dst = LinearRS(self.irreps_dst, self.irreps_emb, bias=dst_bias)


        if attn_type not in ['mlp', 'linear', 'dp']:
            raise ValueError(f"Unknown attention type: {attn_type}")
        self.attn_type: str = attn_type
        if self.attn_type == 'mlp':
            self.ga = GraphAttentionMLP(irreps_emb = self.irreps_emb,
                                        irreps_edge_attr = self.irreps_edge_attr,
                                        irreps_node_output = self.irreps_dst,
                                        fc_neurons = self.fc_neurons,
                                        irreps_head = self.irreps_head,
                                        num_heads=self.num_heads, 
                                        alpha_drop=alpha_drop, 
                                        proj_drop=proj_drop)
        elif self.attn_type == 'linear':
            raise NotImplementedError
        elif self.attn_type == 'dp':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")
        

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        self.norm_2 = EquivariantLayerNormV2(self.irreps_dst)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_dst, 
            irreps_node_output=self.irreps_dst, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
            
    def forward(self, node_input_src: torch.Tensor,
                node_input_dst: torch.Tensor,
                batch_dst: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_scalars: torch.Tensor) -> torch.Tensor:

        message_src: torch.Tensor = self.norm_1_src(node_input_src)
        message_src: torch.Tensor = self.linear_src(node_input_src)

        message_dst: torch.Tensor = self.norm_1_dst(node_input_dst)
        message_dst: torch.Tensor = self.linear_dst(node_input_dst)

        message: torch.Tensor = message_src[edge_src] + message_dst[edge_dst]
        
        node_features: torch.Tensor = self.ga(message=message, 
                                              edge_dst=edge_dst, 
                                              edge_attr=edge_attr, 
                                              edge_scalars=edge_scalars,
                                              n_nodes_dst = len(node_input_dst))
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch_dst)
        node_output: torch.Tensor = node_input_dst + node_features # skip connection
        
        node_features: torch.Tensor = self.norm_2(node_output, batch=batch_dst)
        node_features: torch.Tensor = self.ffn(node_features)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch_dst)
        node_output = node_output + node_features
        
        return node_output


@compile_mode('script')
class PoolingBlock(torch.nn.Module):  
    def __init__(self,
        irreps_src: o3.Irreps, 
        irreps_dst: o3.Irreps, 
        irreps_edge_attr: o3.Irreps, 
        irreps_head: o3.Irreps,
        num_heads: int, 
        fc_neurons: List[int],
        pool_radius: float,
        pool_ratio: float,
        pool_method: str = 'fps',
        deterministic: bool = False,
        irreps_mlp_mid: Union[o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0):
        
        super().__init__()
        self.irreps_src: o3.Irreps = o3.Irreps(irreps_src)
        self.irreps_dst: o3.Irreps = o3.Irreps(irreps_dst)
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons

        if self.irreps_src != self.irreps_dst:
            raise NotImplementedError

        self.block = EquiformerBlock(irreps_src = self.irreps_src, 
                                     irreps_dst = self.irreps_dst, 
                                     irreps_edge_attr = self.irreps_edge_attr, 
                                     irreps_head = self.irreps_head,
                                     num_heads = self.num_heads, 
                                     fc_neurons = self.fc_neurons,
                                     irreps_mlp_mid = irreps_mlp_mid,
                                     attn_type = attn_type,
                                     alpha_drop = alpha_drop,
                                     proj_drop = proj_drop,
                                     drop_path_rate = drop_path_rate,
                                     src_bias = False,
                                     dst_bias = True)
        
        self.pool_radius: float = pool_radius
        self.pool_ratio: float = pool_ratio
        assert isinstance(pool_method, str), f"Unknown pooling method: {pool_method}"
        if pool_method == 'fps':
            self.pool_layer = FpsPool(ratio=self.pool_ratio, random_start=not deterministic, r=self.pool_radius, max_num_neighbors=1000)
        else:
            raise ValueError(f"Unknown pooling method: {pool_method}")


        assert len(fc_neurons) >= 1
        self.num_radial_basis = fc_neurons[0]
        self.radial_basis_fn = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.num_radial_basis, cutoff=self.pool_radius * 0.99)

        self.spherical_harmonics = o3.SphericalHarmonics(irreps_out = self.irreps_edge_attr, normalize = True, normalization='component')

    def forward(self, node_feature: torch.Tensor,
                node_coord: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        node_coord_src = node_coord
        node_feature_src = node_feature
        batch_src = batch

        node_feature_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst = self.pool_layer(node_coord_src = node_coord_src, 
                                                                                                  node_feature_src = node_feature_src, 
                                                                                                  batch_src = batch_src)
        
        edge_vec: torch.Tensor = node_coord_src.index_select(0, edge_src) - node_coord_dst.index_select(0, edge_dst)
        edge_attr = self.spherical_harmonics(edge_vec)
        edge_length = edge_vec.norm(dim=1, p=2)
        edge_scalars = self.radial_basis_fn(edge_length)

        node_feature_dst = self.block(node_input_src = node_feature_src,
                                      node_input_dst = node_feature_dst,
                                      batch_dst = batch_dst,
                                      edge_src = edge_src,
                                      edge_dst = edge_dst,
                                      edge_attr = edge_attr,
                                      edge_scalars = edge_scalars)
        
        return node_feature_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst

