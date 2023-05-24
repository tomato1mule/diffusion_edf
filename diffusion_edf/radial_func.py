import warnings
import math
from typing import Union, Optional, List, Tuple, Dict
from beartype import beartype

import torch
import torch.nn.functional as F

@torch.jit.script
def gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # pi = 3.14159
    # a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) # / (a * std)

@torch.jit.script
def soft_step(x, n: int = 3):
    return (x>0) * ((x<1)*((n+1)*x.pow(n)-n*x.pow(n+1)) + (x>=1))

@torch.jit.script
def soft_cutoff(x, thr:float = 0.8, n:int = 3):
    x = (x-thr) / (1-thr)
    return 1-soft_step(x, n=n)

@torch.jit.script
def soft_square_cutoff(x, thr:float = 0.8, n:int = 3, infinite: bool = False) -> torch.Tensor:
    if infinite:
        return soft_cutoff(x, thr=thr, n=n) * (x > 0.5) + soft_cutoff(1-x, thr=thr, n=n) * (x <= 0.5)
    else:
        return (x > 0.5) + soft_cutoff(1-x, thr=thr, n=n) * (x <= 0.5)
    
@torch.jit.script
def soft_square_cutoff_2(x: torch.Tensor, ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]], n:int = 3) -> torch.Tensor:
    """
    Input:
        ranges: (left_end, left_begin, right_begin, right_end)
        n: n-th polynomial is used.
    """
    if ranges is None:
        return x
    
    assert len(ranges) == 4
    left_end, left_begin, right_begin, right_end = ranges
    if left_end is None or left_begin is None:
        assert left_end is None and left_begin is None
        div_l: float = 1.
    else:
        div_l: float = left_begin - left_end

    if right_end is None or right_begin is None:
        assert right_end is None and right_begin is None
        div_r: float = 1.
    else:
        div_r: float = right_end - right_begin

    
    if right_begin is not None and left_end is None:
        y = 1-soft_step((x-right_begin) / div_r, n=n)
    elif left_end is not None and right_begin is None:
        y = soft_step((x-left_end) / div_l, n=n)
    elif right_begin is not None and left_end is not None and left_begin is not None:
        assert left_begin <= right_begin
        y = (1-soft_step((x-right_begin) / div_r, n=n)) * (x>0.5*(left_begin+right_begin)) + soft_step((x-left_end) / div_l, n=n) * (x<=0.5*(left_begin+right_begin))
    else:
        y = torch.ones_like(x)

    

    return y

class BesselBasisEncoder(torch.nn.Module):
    def __init__(self, n_basis: int, start: Optional[float] = 0., end: float = 1., cutoff: bool = False) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.c = end - start
        self.n_basis = n_basis
        self.register_buffer('bessel_roots', torch.arange(1, self.n_basis + 1) * math.pi)
        self.sqrt_two_div_c = math.sqrt(2 / self.c)
        self.cutoff = cutoff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., None] - self.start
        x_div_c = x / self.c
        out = self.sqrt_two_div_c * torch.sin(self.bessel_roots * x_div_c) / x
        if not self.cutoff:
            return out
        else:
            return out * (x_div_c < 1) * (0 < x)



class GaussianRadialBasisLayerFiniteCutoff(torch.nn.Module):
    def __init__(self, num_basis: int, cutoff: float, soft_cutoff: bool = True, offset: Optional[float] = None, cutoff_thr_ratio: float = 0.8, infinite: bool = False):
        super().__init__()
        self.num_basis: int = num_basis
        self.cutoff: float = float(cutoff)
        if offset is None:
            offset = 0.01 * self.cutoff # For stability, weights should be zero when edge length is very small (otherwise, gradients of spherical harmonics would blow up).
        self.offset: float = float(offset)
        if self.offset < 0.:
            warnings.warn(f"Negative offset ({self.offset}) is provided for radial basis encoder. Are you sure?")

        self.mean_init_max = 1.0
        self.mean_init_min = 0
        mean = torch.linspace(self.mean_init_min, self.mean_init_max, self.num_basis+2)[1:-1].unsqueeze(0)
        self.mean = torch.nn.Parameter(mean)

        self.std_logit = torch.nn.Parameter(torch.zeros(1, self.num_basis))        # Softplus logit
        self.weight_logit = torch.nn.Parameter(torch.zeros(1, self.num_basis))     # Sigmoid logit

        init_std = 2.0 / self.num_basis
        torch.nn.init.constant_(self.std_logit, math.log(math.exp((init_std)) -1)) # Inverse Softplus

        self.max_weight = 4.
        torch.nn.init.constant_(self.weight_logit, -math.log(self.max_weight/1. - 1)) # Inverse Softplus

        self.soft_cutoff: bool = soft_cutoff
        self.cutoff_thr_ratio: float = cutoff_thr_ratio
        assert cutoff_thr_ratio <= 0.95

        self.normalizer = math.sqrt(self.num_basis)
        self.infinite = infinite
        

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = (dist - self.offset) / (self.cutoff - self.offset)
        dist = dist.unsqueeze(-1)

        
        x = dist.expand(-1, self.num_basis)
        mean = self.mean
        std = F.softplus(self.std_logit) + 1e-5
        x = gaussian(x, mean, std)
        x = torch.sigmoid(self.weight_logit) * self.max_weight * x

        if self.soft_cutoff is True:
            x = x * soft_square_cutoff(dist, thr=self.cutoff_thr_ratio, infinite=self.infinite)
        
        return x * self.normalizer
    
    
    # def extra_repr(self):
    #     return 'mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}'.format(
    #         self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min)






# Borrowed from https://github.com/lucidrains/denoising-diffusion-pytorch
class SinusoidalPositionEmbeddings(torch.nn.Module):
    """
    dim: Output encoder dimension
    max_val: input assumed to be in 0~max_val
    n: The period of each sinusoidal kernel ranges from 2pi~n*2pi
    """
    @beartype
    def __init__(self, dim: int, max_val: float, n: float = 10000.):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, f"dim must be an even number!"
        self.n = n
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.max_val * self.n # time: 0~10000

        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(self.n) / (half_dim - 1) # Period: 2pi~10000*2pi
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # shape: (self.dim/2, )
        embeddings = x[..., None] * embeddings                                      # shape: (*x.shape, self.dim/2) 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)        # shape: (*x.shape, self.dim)

        return embeddings # (*x.shape, self.dim)