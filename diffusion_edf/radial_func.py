import math

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
def soft_cutoff(x, thr:float = 0.8, n:int = 3, offset:float = 0.):
    x = (x-thr) / (1-thr)
    x = x*(1+offset)
    return 1-soft_step(x, n=n)

@torch.jit.script
def soft_square_cutoff(x, thr:float = 0.8, n:int = 3) -> torch.Tensor:
    return soft_cutoff(x, thr=thr, n=n, offset=0.) * (x > 0.5) + soft_cutoff(1-x, thr=thr, n=n, offset=0.) * (x <= 0.5)



class GaussianRadialBasisLayerFiniteCutoff(torch.nn.Module):
    def __init__(self, num_basis: int, cutoff: float, soft_cutoff: bool = True):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0

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
        

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist / self.cutoff
        dist = dist.unsqueeze(-1)

        
        x = dist.expand(-1, self.num_basis)
        mean = self.mean
        std = F.softplus(self.std_logit) + 1e-5
        x = gaussian(x, mean, std)
        x = torch.sigmoid(self.weight_logit) * self.max_weight * x

        if self.soft_cutoff is True:
            x = x * soft_square_cutoff(dist)

        return x
    
    
    def extra_repr(self):
        return 'mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}'.format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min)
    