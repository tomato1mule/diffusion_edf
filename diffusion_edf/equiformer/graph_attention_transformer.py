from typing import List, Tuple, Any, Dict, Union, Optional

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

# import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
                                     FullyConnectedTensorProductRescale, 
                                     FullyConnectedTensorProductRescaleSwishGate, 
                                     DepthwiseTensorProduct,
                                     irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate, SmoothLeakyReLU
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath

# for bessel radial basis
# from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from .bessel_rbf import RadialBasis




def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))
            

def get_mul_0(irreps: o3.Irreps) -> int:
    mul_0: int = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


#@compile_mode('script')
class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.
        
        DTP + RadialFC + Linear (+ LayerNorm + Gate)

        Parameters
        ----------
        fc_neurons : list of function or None
            list of activation functions, `None` if non-scalar or identity
    '''
    def __init__(self, irreps_node_input: o3.Irreps, irreps_edge_attr: o3.Irreps, irreps_node_output: o3.Irreps, 
        fc_neurons: Optional[List[int]], use_activation: bool = False, norm_layer: Optional[str] = None, 
        internal_weights: bool = False):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        norm = get_norm_layer(norm_layer)

        self.dtp: TensorProductRescale = DepthwiseTensorProduct(self.irreps_node_input, 
                                                                self.irreps_edge_attr, 
                                                                self.irreps_node_output, 
                                                                bias=False, 
                                                                internal_weights=internal_weights)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel]) # Simple Linear layer for radial function. Each layer dim is: [fc_neuron1 (input), fc_neuron2, ..., weight_numel (output)]
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():  # Seems to be for normalization
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k # Seems to be for normalization
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k            # Seems to be for normalization 
                
        irreps_lin_output: o3.Irreps = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output: o3.Irreps = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output: o3.Irreps = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)
        
        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0: # use typical scalar activation if irreps_out is all scalar (L=0)
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU() for _ in self.irreps_node_output])
            else: # use gate nonlinearity if there are non-scalar (L>0) components in the irreps_out.
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _ in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _ in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input: torch.Tensor, edge_attr: torch.Tensor, edge_scalars: Optional[torch.Tensor], 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor: # Batch does nothing if you use EquivLayernormV2
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
        '''
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        else:
            weight = None
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        

#@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''
    def __init__(self, irreps_head: o3.Irreps, num_heads: int):
        super().__init__()
        self.num_heads: int = num_heads
        self.irreps_head: o3.Irreps = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
        self.mid_in_indices = tuple(self.mid_in_indices)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _ = x.shape
        out = []
        for start_idx, end_idx in self.mid_in_indices:
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, temp.shape[-1] // self.num_heads)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)
    
    
#@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''
    def __init__(self, irreps_head: o3.Irreps):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
        self.head_indices = tuple(self.head_indices)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, _ = x.shape
        out = []
        for start_idx, end_idx in self.head_indices:
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


# class ConcatIrrepsTensor(torch.nn.Module):
    
#     def __init__(self, irreps_1, irreps_2):
#         super().__init__()
#         assert irreps_1 == irreps_1.simplify()
#         self.check_sorted(irreps_1)
#         assert irreps_2 == irreps_2.simplify()
#         self.check_sorted(irreps_2)
        
#         self.irreps_1 = irreps_1
#         self.irreps_2 = irreps_2
#         self.irreps_out = irreps_1 + irreps_2
#         self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out) #self.irreps_out.sort()
#         self.irreps_out = self.irreps_out.simplify()
        
#         self.ir_mul_list = []
#         lmax = max(irreps_1.lmax, irreps_2.lmax)
#         irreps_max = []
#         for i in range(lmax + 1):
#             irreps_max.append((1, (i, -1)))
#             irreps_max.append((1, (i,  1)))
#         irreps_max = o3.Irreps(irreps_max)
        
#         start_idx_1, start_idx_2 = 0, 0
#         dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
#         for _, ir in irreps_max:
#             dim_1, dim_2 = None, None
#             index_1 = self.get_ir_index(ir, irreps_1)
#             index_2 = self.get_ir_index(ir, irreps_2)
#             if index_1 != -1:
#                 dim_1 = dim_1_list[index_1]
#             if index_2 != -1:
#                 dim_2 = dim_2_list[index_2]
#             self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
#             start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
#             start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2
          
            
#     def get_irreps_dim(self, irreps):
#         muls = []
#         for mul, ir in irreps:
#             muls.append(mul * ir.dim)
#         return muls
    
    
#     def check_sorted(self, irreps):
#         lmax = None
#         p = None
#         for _, ir in irreps:
#             if p is None and lmax is None:
#                 p = ir.p
#                 lmax = ir.l
#                 continue
#             if ir.l == lmax:
#                 assert p < ir.p, 'Parity order error: {}'.format(irreps)
#             assert lmax <= ir.l                
        
    
#     def get_ir_index(self, ir, irreps):
#         for index, (_, irrep) in enumerate(irreps):
#             if irrep == ir:
#                 return index
#         return -1
    
    
#     def forward(self, feature_1, feature_2):
        
#         output = []
#         for i in range(len(self.ir_mul_list)):
#             start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
#             if mul_1 is not None:
#                 output.append(feature_1.narrow(-1, start_idx_1, mul_1))
#             if mul_2 is not None:
#                 output.append(feature_2.narrow(-1, start_idx_2, mul_2))
#         output = torch.cat(output, dim=-1)
#         return output
    
    
#     def __repr__(self):
#         return '{}(irreps_1={}, irreps_2={})'.format(self.__class__.__name__, 
#             self.irreps_1, self.irreps_2)

        
#@compile_mode('script')
class GraphAttentionMLP(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
        irreps_node_input: o3.Irreps, irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps, irreps_node_output: o3.Irreps,
        fc_neurons: Optional[List[int]],
        irreps_head: o3.Irreps, num_heads: int, 
        irreps_pre_attn: Optional[o3.Irreps] = None, 
        rescale_degree: bool = False, 
        alpha_drop: float = 0.1, proj_drop: float = 0.1,
        src_bias: bool = True, dst_bias: bool = False):
        
        super().__init__()
        self.irreps_node_input: o3.Irreps = o3.Irreps(irreps_node_input)
        self.irreps_node_attr: o3.Irreps = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output: o3.Irreps = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn: o3.Irreps = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        self.rescale_degree: bool = rescale_degree
        
        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=src_bias)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=dst_bias)
        
        irreps_attn_heads: o3.Irreps = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads: o3.Irreps = irreps_attn_heads.simplify() 
        mul_alpha: int = get_mul_0(irreps_attn_heads) # how many 0e in irreps_attn_heads
        mul_alpha_head: int = mul_alpha // num_heads  # how many 0e per head
        irreps_alpha: o3.Irreps = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all: o3.Irreps = (irreps_alpha + irreps_attn_heads).simplify()
        
        # Use an extra separable FCTP and Swish Gate for value
        self.sep_act = SeparableFCTP(irreps_node_input = self.irreps_pre_attn, 
                                        irreps_edge_attr = self.irreps_edge_attr, 
                                        irreps_node_output = self.irreps_pre_attn, 
                                        fc_neurons = fc_neurons, 
                                        use_activation = True, 
                                        norm_layer = None, 
                                        internal_weights = False)
        self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
        self.sep_value = SeparableFCTP(irreps_node_input = self.irreps_pre_attn, 
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
            self.proj_drop = EquivariantDropout(irreps = self.irreps_node_input, 
                                                drop_prob = proj_drop)
        
        
    def forward(self, node_input: torch.Tensor, 
                edge_src: torch.Tensor, edge_dst: torch.Tensor, 
                edge_attr: torch.Tensor, edge_scalars: torch.Tensor) -> torch.Tensor:
        
        message_src: torch.Tensor = self.merge_src(node_input)
        message_dst: torch.Tensor = self.merge_dst(node_input)
        message: torch.Tensor = message_src[edge_src] + message_dst[edge_dst]
        
      
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
        attn: torch.Tensor = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn: torch.Tensor = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree: torch.Tensor = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree: torch.Tensor = degree.view(-1, 1)
            attn = attn * degree
            
        node_output: torch.Tensor = self.proj(attn) # Final Linear layer.
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super().extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str


#@compile_mode('script')
class GraphAttentionLinear(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
        irreps_node_input: o3.Irreps, irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps, irreps_node_output: o3.Irreps,
        fc_neurons: Optional[List[int]],
        irreps_head: o3.Irreps, num_heads: int, 
        irreps_pre_attn: Optional[o3.Irreps] = None, 
        rescale_degree: bool = False,
        alpha_drop: float = 0.1, proj_drop: float = 0.1,
        src_bias: bool = True, dst_bias: bool = False):
        
        super().__init__()
        self.irreps_node_input: o3.Irreps = o3.Irreps(irreps_node_input)
        self.irreps_node_attr: o3.Irreps = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output: o3.Irreps = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn: o3.Irreps = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        self.rescale_degree: bool = rescale_degree
        
        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=src_bias)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=dst_bias)
        
        irreps_attn_heads: o3.Irreps = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads: o3.Irreps = irreps_attn_heads.simplify() 
        mul_alpha: int = get_mul_0(irreps_attn_heads) # how many 0e in irreps_attn_heads
        mul_alpha_head: int = mul_alpha // num_heads  # how many 0e per head
        irreps_alpha: o3.Irreps = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all: o3.Irreps = (irreps_alpha + irreps_attn_heads).simplify()
        

        self.sep = SeparableFCTP(irreps_node_input = self.irreps_pre_attn, 
                                    irreps_edge_attr = self.irreps_edge_attr, 
                                    irreps_node_output = irreps_attn_all, 
                                    fc_neurons = fc_neurons, 
                                    use_activation = False, 
                                    norm_layer = None,
                                    internal_weights = False)
        self.vec2heads = Vec2AttnHeads(irreps_head = (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), 
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
            self.proj_drop = EquivariantDropout(irreps = self.irreps_node_input, 
                                                drop_prob = proj_drop)
        
        
    def forward(self, node_input: torch.Tensor, 
                edge_src: torch.Tensor, edge_dst: torch.Tensor, 
                edge_attr: torch.Tensor, edge_scalars: torch.Tensor) -> torch.Tensor:
        
        message_src: torch.Tensor = self.merge_src(node_input)
        message_dst: torch.Tensor = self.merge_dst(node_input)
        message: torch.Tensor = message_src[edge_src] + message_dst[edge_dst]
        

        # No Gate -> DTP -> Linear as in MLP attention.
        message: torch.Tensor = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
        message: torch.Tensor = self.vec2heads(message)
        head_dim_size: int = message.shape[-1]
        alpha: torch.Tensor = message.narrow(-1, 0, self.mul_alpha_head)
        value: torch.Tensor = message.narrow(-1, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        # inner product
        alpha: torch.Tensor = self.alpha_act(alpha)          # Leaky ReLU
        alpha: torch.Tensor = torch.einsum('ehk, hk -> eh', alpha, self.alpha_dot.squeeze(0)) # Linear layer: (N_edge, N_head mul_alpha_head) -> (N_edge, N_head)
        alpha: torch.Tensor = torch_geometric.utils.softmax(alpha, edge_dst, dim=-2)          # Softmax
        alpha: torch.Tensor = alpha.unsqueeze(-1)                              # (N_edge, N_head)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn: torch.Tensor = value * alpha
        attn: torch.Tensor = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn: torch.Tensor = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree: torch.Tensor = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree: torch.Tensor = degree.view(-1, 1)
            attn = attn * degree
            
        node_output: torch.Tensor = self.proj(attn) # Final Linear layer.
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super().extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str




#@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
        irreps_node_input: o3.Irreps, irreps_node_attr: o3.Irreps,
        irreps_node_output: o3.Irreps, irreps_mlp_mid: Optional[o3.Irreps] = None,
        proj_drop: float = 0.1, bias: bool = True, rescale: bool = True):
        
        super().__init__()
        self.irreps_node_input: o3.Irreps = o3.Irreps(irreps_node_input)
        self.irreps_node_attr: o3.Irreps = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid: o3.Irreps = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output: o3.Irreps = o3.Irreps(irreps_node_output)
        
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
            
        
    def forward(self, node_input: torch.Tensor, node_attr: torch.Tensor) -> torch.Tensor:
        node_output: torch.Tensor = self.fctp_1(node_input, node_attr)
        node_output: torch.Tensor = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output: torch.Tensor = self.proj_drop(node_output)
        return node_output
    
    
#@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''
    
    def __init__(self,
        irreps_node_input: o3.Irreps, 
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps, 
        irreps_node_output: o3.Irreps,
        fc_neurons: Optional[List[int]],
        irreps_head: o3.Irreps,
        num_heads: int, 
        irreps_pre_attn: Optional[o3.Irreps] = None, 
        rescale_degree: bool = False, 
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0,
        irreps_mlp_mid: Optional[o3.Irreps] = None, 
        norm_layer: str = 'layer'):
        
        super().__init__()
        self.irreps_node_input: o3.Irreps = o3.Irreps(irreps_node_input)
        self.irreps_node_attr: o3.Irreps = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr: o3.Irreps = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output: o3.Irreps = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn: o3.Irreps = self.irreps_node_input if irreps_pre_attn is None \
                                                                 else o3.Irreps(irreps_pre_attn)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads
        self.rescale_degree: int = rescale_degree
        if attn_type not in ['mlp', 'linear', 'dp']:
            raise ValueError(f"Unknown attention type: {attn_type}")
        self.attn_type: str = attn_type
        self.irreps_mlp_mid: o3.Irreps = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
                                                                   else self.irreps_node_input
        
        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        if self.attn_type == 'mlp':
            self.ga = GraphAttentionMLP(irreps_node_input=self.irreps_node_input, 
                                    irreps_node_attr=self.irreps_node_attr,
                                    irreps_edge_attr=self.irreps_edge_attr, 
                                    irreps_node_output=self.irreps_node_input,
                                    fc_neurons=fc_neurons,
                                    irreps_head=self.irreps_head, 
                                    num_heads=self.num_heads, 
                                    irreps_pre_attn=self.irreps_pre_attn, 
                                    rescale_degree=self.rescale_degree, 
                                    alpha_drop=alpha_drop, 
                                    proj_drop=proj_drop)
        elif self.attn_type == 'linear':
            self.ga = GraphAttentionLinear(irreps_node_input=self.irreps_node_input, 
                                    irreps_node_attr=self.irreps_node_attr,
                                    irreps_edge_attr=self.irreps_edge_attr, 
                                    irreps_node_output=self.irreps_node_input,
                                    fc_neurons=fc_neurons,
                                    irreps_head=self.irreps_head, 
                                    num_heads=self.num_heads, 
                                    irreps_pre_attn=self.irreps_pre_attn, 
                                    rescale_degree=self.rescale_degree, 
                                    alpha_drop=alpha_drop, 
                                    proj_drop=proj_drop)
        elif self.attn_type == 'dp':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")
        
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        
        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        #self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input, 
        #    self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input, #self.concat_norm_output.irreps_out, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr, 
                self.irreps_node_output, 
                bias=True, rescale=True)
            
            
    def forward(self, node_input: torch.Tensor, node_attr: torch.Tensor, 
                edge_src: torch.Tensor, edge_dst: torch.Tensor, 
                edge_attr: torch.Tensor, edge_scalars: torch.Tensor, 
                batch: Optional[torch.Tensor]) -> torch.Tensor:
        
        node_output: torch.Tensor = node_input
        node_features: torch.Tensor = node_input
        node_features: torch.Tensor = self.norm_1(node_features, batch=batch)
        #norm_1_output = node_features
        node_features: torch.Tensor = self.ga(node_input=node_features, 
                                              edge_src=edge_src, edge_dst=edge_dst, 
                                              edge_attr=edge_attr, edge_scalars=edge_scalars)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output: torch.Tensor = node_output + node_features
        
        node_features: torch.Tensor = node_output
        node_features: torch.Tensor = self.norm_2(node_features, batch=batch)
        #node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features: torch.Tensor = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        return node_output
    

class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num: float):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int, dim_size: Optional[int] = None) -> torch.Tensor:
        out = scatter(x, index, dim=dim, dim_size=dim_size)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)
    

#@compile_mode('script')
class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding: o3.Irreps, irreps_edge_attr: o3.Irreps, fc_neurons: Optional[List[int]], 
                 avg_aggregate_num: float,
                 use_bias: bool = True, 
                 rescale: bool = True):
        super().__init__()
        if fc_neurons is None:
            fc_neurons = []
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
                            bias=use_bias, rescale=rescale)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
        self.rad = RadialProfile(ch_list = fc_neurons + [self.dw.tp.weight_numel]) # Simple Linear layer for radial function. Each layer dim is: [fc_neuron1 (input), fc_neuron2, ..., weight_numel (output)]
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, node_input: torch.Tensor, 
                edge_attr: torch.Tensor, 
                edge_scalars: torch.Tensor, 
                edge_src: torch.Tensor, 
                edge_dst: torch.Tensor) -> torch.Tensor:
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, 
            dim_size=node_features.shape[0])
        return node_features
    



####################################################################
####################################################################
####################################################################
########################### QM9 Specific ###########################
####################################################################
####################################################################
####################################################################


_RESCALE = True

# QM9
_MAX_ATOM_TYPE = 5
# Statistics of QM9 with cutoff radius = 5
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
            self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
        
    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)
        
        return node_embedding, node_attr, node_atom_onehot



class GraphAttentionTransformer(torch.nn.Module):
    def __init__(self,
        irreps_in='5x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            raise NotImplementedError
            #self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE)) 
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer)
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                #or isinstance(module, GaussianRadialBasisLayer) 
                or isinstance(module, RadialBasis)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    def forward(self, f_in, pos, batch, node_atom, **kwargs) -> torch.Tensor:
        
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch,
            max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        edge_length = edge_vec.norm(dim=1)
        #edge_length_embedding = sin_pos_embedding(x=edge_length, 
        #    start=0.0, end=self.max_radius, number=self.number_of_basis, 
        #    cutoff=False)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        outputs = self.head(node_features)
        outputs = self.scale_scatter(outputs, batch, dim=0)
        
        if self.scale is not None:
            outputs = self.scale * outputs

        return outputs


@register_model
def graph_attention_transformer_l2(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2_e3(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+32x0o+32x1e+32x1o+16x2e+16x2o', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+8x0o+8x1e+8x1o+4x2e+4x2o', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+96x0o+96x1e+96x1o+48x2e+48x2o',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.2
@register_model
def graph_attention_transformer_nonlinear_bessel_l2(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.1
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop01(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.1, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.0
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop00(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model