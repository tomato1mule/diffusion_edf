import warnings
from typing import Union, Optional, List, Tuple, Dict
import math

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_cluster import radius
from torch_scatter import scatter_sum

from diffusion_edf.equiformer.graph_attention_transformer import sort_irreps_even_first

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

        self.register_buffer('sign', sign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sign
    

# Borrowed from https://github.com/lucidrains/denoising-diffusion-pytorch
class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim: int, max_t: float = 1., n: float = 10000.):
        super().__init__()
        self.dim = dim
        self.n = n
        self.max_t = max_t

    def forward(self, time: torch.Tensor):
        assert time.ndim == 1, f"{time.shape}"
        time = time / self.max_t * self.n # time: 0~10000

        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.n) / (half_dim - 1) # Period: 2pi~10000*2pi
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # shape: (self.dim/2, )
        embeddings = time[:, None] * embeddings[None, :]                            # shape: (nBatch, self.dim/2) 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)        # shape: (nBatch, self.dim)

        return embeddings