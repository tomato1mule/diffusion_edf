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
    
    if len(ranges) != 4:
        raise ValueError(f"Wrong ranges armument: {ranges}")
    left_end, left_begin, right_begin, right_end = ranges
    if left_end is None or left_begin is None:
        if not (left_end is None and left_begin is None):
            raise ValueError(f"Wrong ranges armument: {ranges}")
        div_l: float = 1.
    else:
        div_l: float = left_begin - left_end

    if right_end is None or right_begin is None:
        if not (right_end is None and right_begin is None):
            raise ValueError(f"Wrong ranges armument: {ranges}")
        div_r: float = 1.
    else:
        div_r: float = right_end - right_begin

    
    if right_begin is not None and left_end is None:
        y = 1-soft_step((x-right_begin) / div_r, n=n)
    elif left_end is not None and right_begin is None:
        y = soft_step((x-left_end) / div_l, n=n)
    elif right_begin is not None and left_end is not None and left_begin is not None:
        if left_begin > right_begin:
            raise ValueError(f"Wrong ranges armument: {ranges}")
        y = (1-soft_step((x-right_begin) / div_r, n=n)) * (x>0.5*(left_begin+right_begin)) + soft_step((x-left_end) / div_l, n=n) * (x<=0.5*(left_begin+right_begin))
    else:
        y = torch.ones_like(x)

    return y

class BesselBasisEncoder(torch.nn.Module):
    min_val: float
    max_val: float
    max_cutoff: bool
    dim: int
    c: float
    sqrt_two_div_c_cube: float
    max_cutoff: float
    dimensionless: bool
    normalize: bool

    @beartype
    def __init__(self, dim: int, 
                 max_val: Union[float, int], 
                 min_val: Union[float, int] = 0., 
                 max_cutoff: bool = False, 
                 dimensionless: bool = True,
                 normalize: bool = False,
                 eps: Union[float, int] = 1e-3) -> None:
        super().__init__()
        self.max_val = float(max_val)
        self.min_val = float(min_val)
        if self.min_val != 0.:
            raise NotImplementedError
        self.c = self.max_val - self.min_val
        self.dim = dim
        self.register_buffer('bessel_roots', torch.arange(1, self.dim + 1) * math.pi)
        self.register_buffer('eps', torch.tensor(float(eps)))
        self.sqrt_two_div_c_cube = math.sqrt(2 / (self.c**3))
        self.max_cutoff = max_cutoff
        self.dimensionless = dimensionless
        self.normalize = normalize
        if self.dim > 10:
            raise ValueError(f"Too may dims for bessel is unstable. Current dim {self.dim}")

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., None] - self.min_val
        x_div_c = (x / self.c)
        if x_div_c.requires_grad:
            x_div_c = torch.where(x_div_c >= self.eps, x_div_c, self.eps + (x_div_c - x_div_c.detach())) # Straight-through gradient estimation trick
        else:
            x_div_c = torch.max(x_div_c, self.eps)
        if self.normalize:
            out = self.bessel_roots * x_div_c
            out = torch.sin(out) / out
        else:
            out = torch.sin(self.bessel_roots * x_div_c) / (x_div_c)
        if not self.dimensionless:
            out = self.sqrt_two_div_c_cube * out

        if not self.max_cutoff:
            return out
        else:
            return out * (x_div_c < 1) # * (0 < x_div_c)
        
class _Deprecated_GaussianRadialBasis(torch.nn.Module):
    @beartype
    def __init__(self, dim: int, max_val: Union[float, int], min_val: Union[float, int] = 0.):
        super().__init__()
        self.dim: int = dim
        self.max_val: float = float(max_val)
        self.min_val: float = float(min_val)
        if self.min_val < 0.:
            warnings.warn(f"Negative min_val ({self.min_val}) is provided for radial basis encoder. Are you sure?")

        self.mean_init_max = 1.0
        self.mean_init_min = 0
        mean = torch.linspace(self.mean_init_min, self.mean_init_max, self.dim+2)[1:-1].unsqueeze(0)
        self.mean = torch.nn.Parameter(mean)

        self.std_logit = torch.nn.Parameter(torch.zeros(1, self.dim))        # Softplus logit
        self.weight_logit = torch.nn.Parameter(torch.zeros(1, self.dim))     # Sigmoid logit

        init_std = 2.0 / self.dim
        torch.nn.init.constant_(self.std_logit, math.log(math.exp((init_std)) -1)) # Inverse Softplus

        self.max_weight = 4.
        torch.nn.init.constant_(self.weight_logit, -math.log(self.max_weight/1. - 1)) # Inverse Softplus

        self.normalizer = math.sqrt(self.dim)
        

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = (dist - self.min_val) / (self.max_val - self.min_val)
        dist = dist.unsqueeze(-1)
        
        x = dist.expand(-1, self.dim)
        mean = self.mean
        std = F.softplus(self.std_logit) + 1e-5
        x = gaussian(x, mean, std)
        x = torch.sigmoid(self.weight_logit) * self.max_weight * x
        
        return x * self.normalizer
    
    
class _GaussianParamModule(torch.nn.Module):
    weight_cap: torch.jit.Final[float]
    
    def __init__(self, dim: int, max_weight: float):
        dim = int(dim) + 0
        max_weight =float(max_weight)
        
        super().__init__()
        self.std_logit = torch.nn.Parameter(
            torch.empty(1, dim, dtype=torch.float32).fill_(
                math.log(math.exp((2.0 / dim)) -1)             # Inverse Softplus
            ), requires_grad=True
        )                                                               # Softplus logit
        
        self.weight_logit = torch.nn.Parameter(
            torch.empty(1, dim, dtype=torch.float32).fill_(
                -math.log(max_weight/1. - 1)                   # Inverse Sigmoid
            ), requires_grad=True
        )                                                               # Sigmoid logit
        
        mean = torch.linspace(0.0, 1.0, dim+2, dtype=torch.float32)[1:-1].unsqueeze(0)
        self.mean = torch.nn.Parameter(mean, requires_grad=True)
        self.weight_cap = max_weight * float(math.sqrt(dim))
        self._detach_out: bool = False
    
    def train(self, mode: bool = True):
        super().train(mode=mode)
        self._detach_out = not mode
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean: torch.Tensor = self.mean + 0.0
        std: torch.Tensor = F.softplus(self.std_logit) + 1e-5
        weight: torch.Tensor = torch.sigmoid(self.weight_logit) * self.weight_cap
        
        if self._detach_out:
            return mean.detach(), std.detach(), weight.detach()
        else:
            return mean, std, weight
    

class GaussianRadialBasis(torch.nn.Module):
    @beartype
    def __init__(self, dim: int, max_val: Union[float, int], min_val: Union[float, int] = 0.):
        super().__init__()
        self.dim: int = int(dim)
        self.max_val: float = float(max_val)
        self.min_val: float = float(min_val)
        if self.min_val < 0.:
            warnings.warn(f"Negative min_val ({self.min_val}) is provided for radial basis encoder. Are you sure?")

        self.param_module = torch.jit.script(_GaussianParamModule(dim=dim, max_weight=4.0))
        

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = (dist.unsqueeze(-1) - self.min_val) / (self.max_val - self.min_val)
        x = dist.expand(-1, self.dim)
        mean, std, weight = self.param_module()

        x = gaussian(x, mean, std)
        return x * weight



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
    def __init__(self, dim: int, max_val: Union[float, int], n: Union[float, int] = 10000.):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, f"dim must be an even number!"
        self.n = float(n)
        self.max_val = float(max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.max_val * self.n # time: 0~10000

        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        embeddings = math.log(self.n) / (half_dim - 1) # Period: 2pi~10000*2pi
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings) # shape: (self.dim/2, )
        embeddings = x[..., None] * embeddings                                      # shape: (*x.shape, self.dim/2) 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)        # shape: (*x.shape, self.dim)

        return embeddings # (*x.shape, self.dim)