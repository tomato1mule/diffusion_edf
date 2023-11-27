from typing import List, Optional, Union, Tuple
import math

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
# from einops import rearrange

from diffusion_edf.equiformer.drop import GraphDropPath, EquivariantDropout
from diffusion_edf.equiformer.tensor_product_rescale import FullyConnectedTensorProductRescale, LinearRS, FullyConnectedTensorProductRescaleSwishGate
from diffusion_edf.equiformer.layer_norm import EquivariantLayerNormV2
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first

from diffusion_edf.graph_attention import GraphAttentionMLP
from diffusion_edf.connectivity import FpsPool, RadiusGraph, RadiusConnect
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff


#@compile_mode('script')
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






#@compile_mode('script')
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
        dst_bias: bool = True,
        dst_feature_layer: bool = True,
        debug: bool = False):
        self.debug = debug
        
        super().__init__()
        self.irreps_src: o3.Irreps = o3.Irreps(irreps_src)
        self.irreps_dst: o3.Irreps = o3.Irreps(irreps_dst)
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons

        self.irreps_emb: o3.Irreps = self.irreps_dst
        assert num_heads*self.irreps_head.dim == self.irreps_emb.dim, f"{num_heads} X {self.irreps_head} != {self.irreps_emb}"
        if isinstance(irreps_mlp_mid, o3.Irreps):
            self.irreps_mlp_mid: o3.Irreps = o3.Irreps(irreps_mlp_mid)
        elif isinstance(irreps_mlp_mid, int):
            self.irreps_mlp_mid = sort_irreps_even_first((self.irreps_emb * irreps_mlp_mid))[0].simplify()

        self.norm_1_src = EquivariantLayerNormV2(self.irreps_src)
        self.linear_src = LinearRS(self.irreps_src, self.irreps_emb, bias=src_bias)

        if dst_feature_layer is True:
            self.dst_feature_layer = True
            self.norm_1_dst = EquivariantLayerNormV2(self.irreps_dst)
            self.linear_dst = LinearRS(self.irreps_dst, self.irreps_emb, bias=dst_bias)
        else:
            self.dst_feature_layer = False
            assert dst_bias is False
            self.norm_1_dst = torch.nn.Identity()
            self.linear_dst = torch.nn.Identity()


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
                                        proj_drop=proj_drop,
                                        debug=self.debug)
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
