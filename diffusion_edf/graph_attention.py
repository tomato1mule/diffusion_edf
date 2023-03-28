from typing import List, Optional, Union, Tuple

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
import torch_geometric
from torch_scatter import scatter

from diffusion_edf.equiformer.tensor_product_rescale import LinearRS
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first, get_mul_0, Vec2AttnHeads, AttnHeads2Vec, SmoothLeakyReLU, SeparableFCTP
from diffusion_edf.equiformer.drop import EquivariantDropout
from diffusion_edf.equiformer.fast_activation import Activation, Gate

@compile_mode('script')
class GraphAttentionMLP(torch.nn.Module):
    def __init__(self,
        irreps_emb: o3.Irreps,
        irreps_edge_attr: o3.Irreps, 
        irreps_node_output: o3.Irreps,
        fc_neurons: List[int],
        irreps_head: o3.Irreps, 
        num_heads: int, 
        alpha_drop: float = 0.1, 
        proj_drop: float = 0.1):
        
        super().__init__()
        self.irreps_emb = o3.Irreps(irreps_emb)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        
        irreps_attn_heads: o3.Irreps = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads: o3.Irreps = irreps_attn_heads.simplify() 
        mul_alpha: int = get_mul_0(irreps_attn_heads) # how many 0e in irreps_attn_heads
        mul_alpha_head: int = mul_alpha // num_heads  # how many 0e per head
        irreps_alpha: o3.Irreps = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        
        # Use an extra separable FCTP and Swish Gate for value
        self.sep_act = SeparableFCTP(irreps_node_input = self.irreps_emb,
                                     irreps_edge_attr = self.irreps_edge_attr, 
                                     irreps_node_output = self.irreps_emb, 
                                     fc_neurons = fc_neurons, 
                                     use_activation = True, 
                                     norm_layer = None, 
                                     internal_weights = False)
        self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
        self.sep_value = SeparableFCTP(irreps_node_input = self.irreps_emb, 
                                       irreps_edge_attr = self.irreps_edge_attr, 
                                       irreps_node_output = irreps_attn_heads, 
                                       fc_neurons = None, 
                                       use_activation = False, 
                                       norm_layer = None, 
                                       internal_weights = True)
        self.vec2heads_alpha = Vec2AttnHeads(irreps_head = o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                                             num_heads = num_heads)
        self.vec2heads_value = Vec2AttnHeads(irreps_head = self.irreps_head, 
                                             num_heads = num_heads)
        
        self.alpha_act = Activation(irreps_in = o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                                    acts = [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head = irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        self.proj = LinearRS(irreps_in = irreps_attn_heads, 
                             irreps_out = self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(irreps = self.irreps_emb, 
                                                drop_prob = proj_drop)
        
        
    def forward(self, message: torch.Tensor,
                edge_dst: torch.Tensor, 
                edge_attr: torch.Tensor, 
                edge_scalars: torch.Tensor,
                n_nodes_dst: int) -> torch.Tensor:
      
        weight: torch.Tensor = self.sep_act.dtp_rad(edge_scalars)
        message: torch.Tensor = self.sep_act.dtp(message, edge_attr, weight)
        alpha: torch.Tensor = self.sep_alpha(message)                        # f_ij^(L=0) part  ||  Linear: irreps_in -> 'mul_alpha x 0e'
        alpha: torch.Tensor = self.vec2heads_alpha(alpha)                    # reshape (N, Heads*head_dim) -> (N, Heads, head_dim)
        value: torch.Tensor = self.sep_act.lin(message)                      # f_ij^(L>=0) part (before activation)
        value: torch.Tensor = self.sep_act.gate(value)                       # f_ij^(L>=0) part (after activation)
        value: torch.Tensor = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars) # DTP + Linear for f_ij^(L>=0) part
        value: torch.Tensor = self.vec2heads_value(value)                    # reshape (N, Heads*head_dim) -> (N, Heads, head_dim)
        
        # inner product
        alpha: torch.Tensor = self.alpha_act(alpha)          # Leaky ReLU
        alpha: torch.Tensor = torch.einsum('ehk, hk -> eh', alpha, self.alpha_dot.squeeze(0)) # Linear layer: (N_edge, N_head mul_alpha_head) -> (N_edge, N_head)
        alpha: torch.Tensor = torch_geometric.utils.softmax(alpha, edge_dst, dim=-2)          # Softmax
        alpha: torch.Tensor = alpha.unsqueeze(-1)                              # (N_edge, N_head)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn: torch.Tensor = value * alpha
        attn: torch.Tensor = scatter(attn, index=edge_dst, dim=0, dim_size=n_nodes_dst)
        attn: torch.Tensor = self.heads2vec(attn)
            
        node_output: torch.Tensor = self.proj(attn) # Final Linear layer.
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super().extra_repr()
        return output_str