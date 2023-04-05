from typing import Union, Optional, List, Tuple, Dict
import math

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
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)