import warnings
from typing import List, Tuple, Dict, Optional, Union, Callable
import math
import itertools

import torch
from e3nn import o3
from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first


def irreps_to_list(irreps: o3.Irreps) -> List[Tuple[int, Tuple[int, int]]]:
    return [(n,(l,p)) for n,(l,p) in irreps]

def irreps_to_tuple(irreps: o3.Irreps) -> Tuple[Tuple[int, Tuple[int, int]]]:
    return tuple(irreps_to_list(irreps=irreps))

def multiply_irreps(irreps: Union[o3.Irreps, str], mult: int, strict: bool = True) -> o3.Irreps:
    assert isinstance(irreps, o3.Irreps) or isinstance(irreps, o3.Irreps)

    output = []
    for mul, ir in irreps:
        if round(mul*mult) != mul*mult and strict is True:
            raise ValueError(f"{irreps} cannot be multiplied by {mult}")
        output.append((round(mul*mult), ir))
    output = o3.Irreps(output)

    return output

#@compile_mode('script')
class ParityInversionSh(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        sign = []
        for mul, (l,_) in self.irreps:
            if l % 2 == 0:
                sign.append(
                    torch.ones((2*l+1)*mul)
                )
            elif l % 2 == 1:
                sign.append(
                    -torch.ones((2*l+1)*mul)
                )
            else:
                raise ValueError(f"unknown degree {l}")
        sign = torch.cat(sign, dim=-1)

        self.register_buffer('sign', sign, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sign


class IrrepwiseDotProduct(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.irreps_out = o3.Irreps([(1,(0,p)) for n,(l,p) in irreps])

        self.input_dim = self.irreps.dim
        self.output_dim = self.irreps_out.dim

        instructions = []
        for i, (n, (l,p)) in enumerate(self.irreps):
            i_1 = i
            i_2 = i
            i_out = i
            mode = 'uuw'
            train = False
            path_weight = 2*l+1
            instruction = (i_1, i_2, i_out, mode, train, path_weight)
            instructions.append(instruction)


        self.tp = o3.TensorProduct(irreps_in1 = self.irreps,
                                   irreps_in2 = self.irreps,
                                   irreps_out = self.irreps_out,
                                   instructions = instructions,
                                   irrep_normalization = 'none',
                                   path_normalization = 'none', #https://github.com/e3nn/e3nn/discussions/385
                                   internal_weights = True,
                                   shared_weights = True)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == y.shape[-1] == self.input_dim 
        return self.tp(x, y)
    
    @staticmethod
    def test() -> bool:
        irreps_list = [o3.Irreps("3x1e+10x0e+5x2e"), o3.Irreps("3x0e+10x0e+5x1e"), o3.Irreps('1x0e')]
        result = True
        for irreps in irreps_list:
            jits = ['default', 'jit']
            for jit in jits:
                if jit == 'default':
                    dp = IrrepwiseDotProduct(irreps=irreps)
                elif jit == 'jit':
                    dp = torch.jit.script(IrrepwiseDotProduct(irreps=irreps))
                else:
                    raise ValueError(f"Unknown jit type {jit}")
                
                contexts = ['same_shape', 'different_shape']
                for context in contexts:
                    if context == 'same_shape':
                        x,y = irreps.randn(5,4,3,2,-1, device = dp.tp.weight.device, dtype = dp.tp.weight.dtype), irreps.randn(5,4,3,2,-1, device = dp.tp.weight.device, dtype = dp.tp.weight.dtype)
                    elif context == 'different_shape':
                        x,y = irreps.randn(5,4,3,2,-1, device = dp.tp.weight.device, dtype = dp.tp.weight.dtype), irreps.randn(4,3,2,-1, device = dp.tp.weight.device, dtype = dp.tp.weight.dtype)

                    z = dp(x,y)
                    z_test = x*y
                    for i in range(len(irreps)):
                        result = torch.isclose(z_test[...,irreps.slices()[i]].sum(dim=-1),    z[...,i], atol=1e-5).type(torch.float32).mean()
                        if result.item() < 0.999:
                            warnings.warn(f"IrrepwiseDotProduct test failed for irreps {irreps} || jit type: {jit} || context: {context} || isclose ratio: {result.item()}")
                            result = False
                            return result
                        else:
                            result = True

        return True
    

def get_irrepwise_scatter_indices(irreps: o3.Irreps) -> torch.Tensor:
    scatter_indices = torch.zeros(irreps.dim, dtype=torch.long)
    for i, slice in enumerate(irreps.slices()):
        scatter_indices[slice] = i
    return scatter_indices

class IrrepwiseApplyScalar(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, binary_ops: Callable):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.irreps_scalar = o3.Irreps([(1,(0,p)) for n,(l,p) in irreps])
        self.irreps_out = self.irreps

        self.input_dim = self.irreps.dim
        self.scalar_dim = self.irreps_scalar.dim
        self.output_dim = self.input_dim
        self.ops = binary_ops

        scatter_indices = get_irrepwise_scatter_indices(irreps=self.irreps)
        self.register_buffer("scatter_indices", scatter_indices, persistent=False)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim and w.shape[-1] == self.scalar_dim
        w = torch.index_select(input=w, dim=-1, index=self.scatter_indices)

        return self.ops(x, w)
    
    @staticmethod
    def test() -> bool:
        irreps_list = [o3.Irreps("3x1e+10x0e+5x2e"), o3.Irreps("3x0e+10x0e+5x1e"), o3.Irreps('1x0e')]
        irreps_scalar_list = [o3.Irreps([(1,(0,p)) for n,(l,p) in irreps]) for irreps in irreps_list]
        jits = ['default', 'jit']
        input_shapes = ['same_shape', 'different_shape']
        ops = [torch.add, torch.mul]

        result = True

        for context in itertools.product(zip(irreps_list, irreps_scalar_list), jits, input_shapes, ops):
            (irreps, irreps_scalar), jit, input_shape, op = context
            module = IrrepwiseApplyScalar(irreps=irreps, binary_ops=op)
            if jit == 'default':
                pass
            elif jit == 'jit':
                module = torch.jit.script(module)
            else:
                raise ValueError(f"Unknown jit type {jit}")
            device = module.scatter_indices.device
            dtype = torch.float32
            
            if input_shape == 'same_shape':
                x,w = irreps.randn(5,4,3,2,-1, device = device, dtype = dtype), irreps_scalar.randn(5,4,3,2,-1, device = device, dtype = dtype)
            elif input_shape == 'different_shape':
                x,w = irreps.randn(5,4,3,2,-1, device = device, dtype = dtype), irreps_scalar.randn(4,3,2,-1, device = device, dtype = dtype)
            else:
                raise ValueError(f"Unknown input_shape type: {input_shape}")

            z = module(x,w)

            scatter_indices = torch.zeros(irreps.dim, dtype=torch.long, device=w.device)
            for i, slice in enumerate(irreps.slices()):
                scatter_indices[slice] = i
            w_test = torch.index_select(input=w, dim=-1, index=scatter_indices)

            z_test = op(x,w_test)
            result = torch.isclose(z, z_test, atol=1e-5).type(torch.float32).mean()
            if result.item() < 0.999:
                warnings.warn(f"IrrepwiseDotProduct test failed for irreps {irreps} || jit type: {jit} || context: {context} || isclose ratio: {result.item()}")
                result = False
                return result
            else:
                result = True

        return result