from typing import Union, Optional, List, Tuple, Dict
import math

import torch
from e3nn import o3

def multiply_irreps(irreps: Union[o3.Irreps, str], mult: int, strict: bool = True) -> o3.Irreps:
    assert isinstance(irreps, o3.Irreps) or isinstance(irreps, o3.Irreps)

    output = []
    for mul, ir in irreps:
        if round(mul*mult) != mul*mult and strict is True:
            raise ValueError(f"{irreps} cannot be multiplied by {mult}")
        output.append((round(mul*mult), ir))
    output = o3.Irreps(output)

    return output

@torch.jit.script
def cutoff_irreps(f: torch.Tensor,
                  edge_cutoff: Optional[torch.Tensor],
                  cutoff_scalar: Optional[torch.Tensor], 
                  cutoff_nonscalar: Optional[torch.Tensor], 
                  irreps: List[Tuple[int, Tuple[int, int]]],
                  log: bool = False) -> torch.Tensor:
    if edge_cutoff is None and cutoff_scalar is None and cutoff_nonscalar is None:
        return f
    
    f_cutoff = []
    last_idx = 0
    for n, (l,p) in irreps:
        d = n * (2*l + 1)
        if l == 0 and cutoff_scalar is not None:
            if log is True:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * torch.exp(cutoff_scalar[..., None])
                )
            else:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * cutoff_scalar[..., None]
                )
        elif l != 0 and cutoff_nonscalar is not None:
            if log is True:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * torch.exp(cutoff_nonscalar[..., None])
                )
            else:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * cutoff_nonscalar[..., None]
                )
        else:
            f_cutoff.append(f[..., last_idx: last_idx+d])
        
        last_idx = last_idx + d

    f_cutoff = torch.cat(f_cutoff, dim=-1)

    if edge_cutoff is not None:
        if log is True:
            f_cutoff = f_cutoff * torch.exp(edge_cutoff[..., None])
        else:
            f_cutoff = f_cutoff * edge_cutoff[..., None]

    return f_cutoff