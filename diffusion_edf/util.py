from typing import Union, Optional, List, Tuple, Dict

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

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

@compile_mode('script')
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