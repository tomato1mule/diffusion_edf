from typing import List, Optional, Union, Tuple
import math
from beartype import beartype

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from diffusion_edf.equiformer.drop import GraphDropPath, EquivariantDropout
from diffusion_edf.equiformer.tensor_product_rescale import FullyConnectedTensorProductRescale, LinearRS, FullyConnectedTensorProductRescaleSwishGate
from diffusion_edf.equiformer.layer_norm import EquivariantLayerNormV2
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first

from diffusion_edf.graph_attention import GraphAttentionMLP2
from diffusion_edf.utils import multiply_irreps
from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge
from diffusion_edf.skip import ProjectIfMismatch


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
    use_src_point_attn: bool
    use_dst_point_attn: bool
    use_edge_weights: bool

    @beartype
    def __init__(self,
        irreps_src: Union[str, o3.Irreps], 
        irreps_dst: Union[str, o3.Irreps], 
        irreps_edge_attr: Union[str, o3.Irreps], 
        num_heads: int, 
        fc_neurons: List[int],
        irreps_emb: Optional[Union[str, o3.Irreps]] = None,
        irreps_output: Optional[Union[str, o3.Irreps]] = None,
        irreps_mlp_mid: Union[o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0,
        use_dst_feature: bool = True,
        skip_connection: bool = True, 
        bias: bool = True,
        use_src_point_attn: bool = False,
        use_dst_point_attn: bool = False,
        use_edge_weights: bool = True):
        
        super().__init__()
        self.irreps_src: o3.Irreps = o3.Irreps(irreps_src)
        self.irreps_dst: o3.Irreps = o3.Irreps(irreps_dst)
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        if irreps_emb is None:
            self.irreps_emb: o3.Irreps = self.irreps_dst
        else:
            self.irreps_emb: o3.Irreps = o3.Irreps(irreps_emb)
        if irreps_output is None:
            self.irreps_output: o3.Irreps = self.irreps_dst
        else:
            self.irreps_output: o3.Irreps = o3.Irreps(irreps_output)
        if isinstance(irreps_mlp_mid, o3.Irreps):
            self.irreps_mlp_mid: o3.Irreps = o3.Irreps(irreps_mlp_mid)
        elif isinstance(irreps_mlp_mid, int):
            self.irreps_mlp_mid = sort_irreps_even_first((self.irreps_emb * irreps_mlp_mid))[0].simplify()
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons
        self.use_dst_feature: bool = use_dst_feature
        if skip_connection is True:
            if self.use_dst_feature is True:
                self.skip_1 = ProjectIfMismatch(irreps_in=self.irreps_dst, irreps_out=self.irreps_emb, bias=True, layernorm=False)
            else:
                self.skip_1 = None
            self.skip_2 = ProjectIfMismatch(irreps_in=self.irreps_emb, irreps_out=self.irreps_output, bias=True, layernorm=False)
        else:
            self.skip_1 = None
            self.skip_2 = None

        if not self.use_dst_feature:
            assert self.skip_1 is None
        self.use_src_point_attn = use_src_point_attn
        self.use_dst_point_attn = use_dst_point_attn
        self.use_edge_weights = use_edge_weights

        if self.use_dst_feature:
            self.prenorm_src = EquivariantLayerNormV2(self.irreps_src, affine=True)
            self.linear_src = LinearRS(self.irreps_src, self.irreps_emb, bias=False)
            self.prenorm_dst = EquivariantLayerNormV2(self.irreps_dst, affine=True)
            self.linear_dst = LinearRS(self.irreps_dst, self.irreps_emb, bias=True)
        else:
            self.prenorm_src = EquivariantLayerNormV2(self.irreps_src, affine=True)
            self.linear_src = LinearRS(self.irreps_src, self.irreps_emb, bias=True)
            self.prenorm_dst = None
            self.linear_dst = None

        if attn_type not in ['mlp', 'linear', 'dp']:
            raise ValueError(f"Unknown attention type: {attn_type}")
        self.attn_type: str = attn_type
        if self.attn_type == 'mlp':
            self.ga = GraphAttentionMLP2(irreps_input = self.irreps_emb,
                                        irreps_edge_attr = self.irreps_edge_attr,
                                        irreps_output = self.irreps_emb,
                                        fc_neurons = self.fc_neurons,
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
        self.post_norm = EquivariantLayerNormV2(self.irreps_emb, affine=bias)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_emb, 
            irreps_node_output=self.irreps_output, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
            
    def forward(self, src_points: FeaturedPoints, 
                dst_points: FeaturedPoints,
                graph_edge: GraphEdge) -> FeaturedPoints:
        assert src_points.x.ndim == 2
        assert dst_points.x.ndim == 2
        
        message_src: torch.Tensor = self.prenorm_src(src_points.f) # Shape: (N_src, F_src)
        message_src: torch.Tensor = self.linear_src(message_src)   # Shape: (N_src, F_emb)

        if self.prenorm_dst is None:
            message_dst = None
        else:
            message_dst = self.prenorm_dst(dst_points.f) # Shape: (N_dst, F_dst)
            if self.linear_dst is not None:
                message_dst = self.linear_dst(message_dst) # Shape: (N_dst, F_emb)
        message: torch.Tensor = message_src[graph_edge.edge_src]         # Shape: (N_edge, F_emb)
        if message_dst is not None:
            message = message + message_dst[graph_edge.edge_dst]         # Shape: (N_edge, F_emb)

        ### Edge Pre Attention (for edge cutoff) ###
        if self.use_edge_weights:
            edge_pre_attn_logit = graph_edge.edge_logits
        else:
            edge_pre_attn_logit = None

        ### Edge Post Attention (for point attention) ###
        if self.use_src_point_attn:
            src_points_w = src_points.w
            assert isinstance(src_points_w, torch.Tensor)
            edge_post_attn = (src_points_w)[graph_edge.edge_src]         # Shape: (N_edge,)
        else:
            edge_post_attn = None
        if self.use_dst_point_attn:
            raise NotImplementedError
        
        emb_features: torch.Tensor = self.ga(message=message, 
                                             graph_edge=graph_edge,
                                             n_nodes_dst = len(dst_points.x),
                                             edge_pre_attn_logit = edge_pre_attn_logit,
                                             edge_post_attn = edge_post_attn) # Shape: (N_dst, F_emb)
        
        if self.drop_path is not None:
            emb_features = self.drop_path(x=emb_features, batch=dst_points.b) # Shape: (N_dst, F_emb)
        if self.skip_1 is not None:
            emb_features = emb_features + self.skip_1(dst_points.f)           # Shape: (N_dst, F_emb)
        
        output_features: torch.Tensor = self.post_norm(emb_features, batch=dst_points.b) # Shape: (N_dst, F_emb)
        output_features: torch.Tensor = self.ffn(output_features) # Shape: (N_dst, F_dst)
        
        if self.drop_path is not None:
            output_features = self.drop_path(output_features, dst_points.b)
        if self.skip_2 is not None:
            output_features = output_features + self.skip_2(emb_features)

        return FeaturedPoints(x=dst_points.x, f=output_features, b=dst_points.b, w=dst_points.w)
