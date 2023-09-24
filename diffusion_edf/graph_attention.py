from typing import List, Optional, Union, Tuple

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_scatter import scatter, scatter_softmax, scatter_logsumexp

from diffusion_edf.equiformer.tensor_product_rescale import LinearRS
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first, get_mul_0, Vec2AttnHeads, AttnHeads2Vec, SmoothLeakyReLU, SeparableFCTP
from diffusion_edf.equiformer.drop import EquivariantDropout
from diffusion_edf.equiformer.fast_activation import Activation, Gate
from diffusion_edf.gnn_data import GraphEdge, FeaturedPoints
from diffusion_edf.irreps_utils import multiply_irreps, cutoff_irreps

#@compile_mode('script')
class GraphAttentionMLP(torch.nn.Module):
    def __init__(self,
        irreps_emb: o3.Irreps,
        irreps_edge_attr: o3.Irreps, 
        irreps_node_output: o3.Irreps,
        fc_neurons: List[int],
        irreps_head: o3.Irreps, 
        num_heads: int, 
        alpha_drop: float = 0.1, 
        proj_drop: float = 0.1,
        debug: bool = False):
        self.debug = debug
        
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
        torch.nn.init.xavier_uniform_(self.alpha_dot) # Following GATv2
        
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
                n_nodes_dst: int,
                edge_attn: Optional[torch.Tensor] = None) -> torch.Tensor:
      
        weight: torch.Tensor = self.sep_act.dtp_rad(edge_scalars)
        message: torch.Tensor = self.sep_act.dtp(message, edge_attr, weight)
        log_alpha = self.sep_alpha(message)                        # f_ij^(L=0) part  ||  Linear: irreps_in -> 'mul_alpha x 0e'
        log_alpha = self.vec2heads_alpha(log_alpha)                    # reshape (N, Heads*head_dim) -> (N, Heads, head_dim)
        value: torch.Tensor = self.sep_act.lin(message)                      # f_ij^(L>=0) part (before activation)
        value: torch.Tensor = self.sep_act.gate(value)                       # f_ij^(L>=0) part (after activation)
        value: torch.Tensor = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars) # DTP + Linear for f_ij^(L>=0) part
        value: torch.Tensor = self.vec2heads_value(value)                    # reshape (N, Heads*head_dim) -> (N, Heads, head_dim)
        # inner product
        log_alpha = self.alpha_act(log_alpha)          # Leaky ReLU
        log_alpha = torch.einsum('ehk, hk -> eh', log_alpha, self.alpha_dot.squeeze(0)) # Linear layer: (N_edge, N_head mul_alpha_head) -> (N_edge, N_head)
        
        # alpha: torch.Tensor = scatter_softmax(log_alpha, edge_dst, dim=-2, dim_size=n_nodes_dst)          # Softmax
        if False: # torch.are_deterministic_algorithms_enabled():
            log_Z = scatter_logsumexp(log_alpha, edge_dst, dim=-2, dim_size = n_nodes_dst) # (NodeNum,1)
        else:
            log_Z = scatter_logsumexp(log_alpha, edge_dst, dim=-2, dim_size = n_nodes_dst) # (NodeNum,1)
        alpha = torch.exp(log_alpha - log_Z[edge_dst]) # (N_edge, N_head)

        alpha: torch.Tensor = alpha.unsqueeze(-1)                              # (N_edge, N_head, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn: torch.Tensor = value * alpha                                     # (N_edge, N_head, head_dim)
        attn: torch.Tensor = scatter(attn, index=edge_dst, dim=0, dim_size=n_nodes_dst)
        attn: torch.Tensor = self.heads2vec(attn)
            
        node_output: torch.Tensor = self.proj(attn) # Final Linear layer.
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super().extra_repr()
        return output_str
    









class GraphAttentionMLP2(torch.nn.Module):
    def __init__(self,
        irreps_input: Union[str, o3.Irreps],
        irreps_edge_attr: Union[str, o3.Irreps], 
        irreps_output: Union[str, o3.Irreps],
        fc_neurons: List[int],
        num_heads: int, 
        irreps_head: Optional[Union[str, o3.Irreps]] = None,
        irreps_mid: Optional[Union[str, o3.Irreps]] = None,
        mul_alpha: Optional[int] = None,
        alpha_drop: float = 0.1, 
        proj_drop: float = 0.1):
        
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        if irreps_mid is None:
            self.irreps_mid = self.irreps_input
        else:
            self.irreps_mid = o3.Irreps(irreps_mid)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_output = o3.Irreps(irreps_output)
        self.num_heads: int = num_heads
        if irreps_head is None:
            self.irreps_head: o3.Irreps = multiply_irreps(self.irreps_mid, 1/self.num_heads, strict=True)
        else:
            self.irreps_head = o3.Irreps(irreps_head)
        
        irreps_attn_heads: o3.Irreps = self.irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads: o3.Irreps = irreps_attn_heads.simplify() 
        self.irreps_attn_heads = irreps_attn_heads
        if mul_alpha is None:
            mul_alpha: int = get_mul_0(irreps_attn_heads) # how many 0e in irreps_attn_heads
        mul_alpha_head: int = mul_alpha // num_heads  # how many 0e per head
        assert mul_alpha_head * num_heads == mul_alpha, f"Head dimension mismatch: {mul_alpha_head} * {num_heads} != {mul_alpha}"
        irreps_alpha: o3.Irreps = o3.Irreps(f'{mul_alpha}x0e') # for attention score
        
        # Use an extra separable FCTP and Swish Gate for value
        self.sep_act = SeparableFCTP(irreps_node_input = self.irreps_input,
                                     irreps_edge_attr = self.irreps_edge_attr, 
                                     irreps_node_output = self.irreps_mid, 
                                     fc_neurons = fc_neurons, 
                                     use_activation = True, 
                                     norm_layer = None, 
                                     internal_weights = False)
        self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
        self.sep_value = SeparableFCTP(irreps_node_input = self.irreps_mid, 
                                       irreps_edge_attr = self.irreps_edge_attr, 
                                       irreps_node_output = irreps_attn_heads, 
                                       fc_neurons = None, 
                                       use_activation = False, 
                                       norm_layer = None, 
                                       internal_weights = True)
        self.vec2heads_alpha = Vec2AttnHeads(irreps_head = o3.Irreps(f'{mul_alpha_head}x0e'), 
                                             num_heads = num_heads)
        self.vec2heads_value = Vec2AttnHeads(irreps_head = self.irreps_head, 
                                             num_heads = num_heads)
        
        self.alpha_act = Activation(irreps_in = o3.Irreps(f'{mul_alpha_head}x0e'), 
                                    acts = [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head = self.irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch.nn.init.xavier_uniform_(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_in = irreps_attn_heads, 
                             irreps_out = self.irreps_output)

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(irreps = self.irreps_output, 
                                                drop_prob = proj_drop)
        

        
    def forward(self, message: torch.Tensor,
                graph_edge: GraphEdge,
                n_nodes_dst: int,
                edge_pre_attn_logit: Optional[torch.Tensor] = None,
                edge_post_attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(graph_edge.edge_attr, torch.Tensor)
        assert isinstance(graph_edge.edge_scalars, torch.Tensor)
        assert message.ndim == 2 # (nEdge, F_in)
        
        edge_scalars = graph_edge.edge_scalars
        edge_attr = graph_edge.edge_attr
        assert edge_scalars is not None and edge_attr is not None # To tell torch.jit.script that it is not None
        
        weight: torch.Tensor = self.sep_act.dtp_rad(edge_scalars)  # (nEdge, numel_1)
        message: torch.Tensor = self.sep_act.dtp(message, edge_attr, weight) # (nEdge, F_pregate)
        log_alpha = self.sep_alpha(message)     # (nEdge, mul_alpha)                                 # f_ij^(L=0) part  ||  Linear: irreps_in -> 'mul_alpha x 0e'
        log_alpha = self.vec2heads_alpha(log_alpha) # (nEdge, nHead, mul_alpha//nHead)               # reshape (N, Heads*head_dim) -> (N, Heads, head_dim)
        value: torch.Tensor = self.sep_act.lin(message) # (nEdge, F_pregate)                         # f_ij^(L>=0) part (before activation)
        value: torch.Tensor = self.sep_act.gate(value) # (nEdge, F_mid)                              # f_ij^(L>=0) part (after activation)
        value: torch.Tensor = self.sep_value(value,                                                  # DTP + Linear for f_ij^(L>=0) part
                                             edge_attr=edge_attr, 
                                             edge_scalars=edge_scalars)  # (nEdge, F_attn)          
        # inner product
        log_alpha = self.alpha_act(log_alpha)                 # (nEdge, nHead, mul_alpha//nHead)         # Leaky ReLU
        log_alpha = torch.einsum('ehk, hk -> eh',             # Linear layer: (N_edge, N_head, mul_alpha//nHead) -> (N_edge, N_head)
                                 log_alpha,                    
                                 self.alpha_dot.squeeze(0))   # (N_edge, N_head)
        if edge_pre_attn_logit is not None:
            log_alpha = log_alpha + edge_pre_attn_logit.unsqueeze(-1)    # For continuity of attention, pre_attn acts on attention logits.
        value: torch.Tensor = self.vec2heads_value(value)                # reshape (nEdge, F_attn) -> (nEdge, nHead, F_attn//nHead)



        # if edge_post_attn is not None:
        #     log_alpha = log_alpha + torch.log(edge_post_attn).unsqueeze(-1)          # (N_edge, N_head)
        if False: # torch.are_deterministic_algorithms_enabled():
            log_Z = scatter_logsumexp(log_alpha, graph_edge.edge_dst, dim=-2, dim_size = n_nodes_dst) # (NodeNum,1)
        else:
            log_Z = scatter_logsumexp(log_alpha, graph_edge.edge_dst, dim=-2, dim_size = n_nodes_dst) # (NodeNum,1)
        alpha = torch.exp(log_alpha - log_Z[graph_edge.edge_dst]) # (N_edge, N_head)
        if edge_post_attn is not None:
            alpha = alpha * edge_post_attn.unsqueeze(-1)          # (N_edge, N_head)

        alpha: torch.Tensor = alpha.unsqueeze(-1)                              # (N_edge, N_head, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)                                  # (N_edge, N_head, 1)
        attn: torch.Tensor = value * alpha                                     # (N_edge, N_head, F_attn//nHead)
        attn: torch.Tensor = scatter(attn, index=graph_edge.edge_dst, dim=0, dim_size=n_nodes_dst) # (N_dst, N_head, F_attn//nHead)
        attn: torch.Tensor = self.heads2vec(attn)                              # (N_dst, F_attn)
            
        node_output: torch.Tensor = self.proj(attn) # (N_dst, F_attn) -> (N_dst, F_out)           # Final Linear layer.
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output) # (N_dst, F_out) 
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super().extra_repr()
        return output_str