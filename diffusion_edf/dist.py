from __future__ import annotations
from typing import Optional, Union, Dict, List, Tuple
import time
import datetime
import os
import random
import math
import warnings

import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_cluster import radius_graph, radius
from torch_scatter import scatter, scatter_logsumexp, scatter_log_softmax
from diffusion_edf import transforms
from xitorch.interpolate import Interp1D




class QuaternionUniformDist(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_symmetric = True

    def sample(self, N=1, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        q = transforms.random_quaternions(N, device = device, dtype=dtype)
        return q

    def propose(self, q: torch.Tensor) -> torch.Tensor:
        N = len(q.view(-1,4))
        q_new = self.sample(N=N, dtype=q.dtype, device=q.device)
        q_new = transforms.quaternion_multiply(q.view(-1,4), q_new)
        return q_new.view(*(q.shape))


@torch.jit.script
def haar_measure_angle(omg: torch.Tensor) -> torch.Tensor:
    assert (omg <= torch.pi).all() and (omg >= 0.).all()
    return (1-torch.cos(omg)) / torch.pi

@torch.jit.script
def haar_measure(q: torch.Tensor) -> torch.Tensor:
    versor = q[..., :0] # cos(omg/2)
    cos_omg = 2 * torch.square(versor) - 1.
    assert (cos_omg <= 1.).all() and (cos_omg >= -1.).all()

    return (1-cos_omg) / torch.pi

@torch.jit.script
def isotropic_gaussian_so3_small_angle(omg: torch.Tensor, eps: Union[float, torch.Tensor]) -> torch.Tensor:
    assert (omg <= torch.pi).all() and (omg >= 0.).all()
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, device=omg.device, dtype=omg.dtype)

    if eps.dtype is torch.float64:
        small_number = 1e-20
        if eps.item() < 1e-10:
            warnings.warn("Too small eps: {eps} is provided.")
    else:
        small_number = 1e-9
        if eps.item() < 1e-5:
            warnings.warn("Too small eps: {eps} is provided. Consider using double precision")

    small_num = small_number / 2 
    small_dnm = (1-torch.exp(-1. * torch.pi**2 / eps)*(2  - 4 * (torch.pi**2) / eps   )) * small_number

    return 0.5 * torch.sqrt(torch.pi) * (eps ** -1.5) * torch.exp((eps - (omg**2 / eps))/4) / (torch.sin(omg/2) + small_num)            \
        * ( small_dnm + omg - ((omg - 2*torch.pi)*torch.exp(torch.pi * (omg - torch.pi) / eps) + (omg + 2*torch.pi)*torch.exp( -torch.pi * (omg+torch.pi) / eps) ))     


def determine_lmax(eps: float) -> int:
    assert eps > 0.
    thr = 10.                                                       # lmax ~= 100 is enough to guarantee exp[-lmax(lmax+1)eps] < exp(-10). for eps = 1e-3                                                                    
    lmax = max(math.ceil(math.sqrt(thr / eps)) , 5)                 # Even for eps = 1e-7, only lmax ~= 10000 is required, which can be calculated almost immediately.
                                                                    # lmax(lmax+1) > lmax^2 >= thr/eps    ---->    exp[-lmax(lmax+1)eps] < exp(-thr).
    return lmax


@torch.jit.script
def isotropic_gaussian_so3_angle(omg: torch.Tensor, eps: Union[float, torch.Tensor], lmax: Optional[int] = None) -> torch.Tensor:
    assert (omg <= torch.pi).all() and (omg >= 0.).all()
    if lmax is None:
        if isinstance(eps, torch.Tensor):
            lmax = determine_lmax(eps=eps.item())
        else:
            lmax = determine_lmax(eps=eps)
        
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, device=omg.device, dtype=omg.dtype)
    
    if eps.dtype is torch.float64:
        small_number = 1e-20
        if eps.item() < 1e-10:
            warnings.warn("Too small eps: {eps} is provided.")
    else:
        small_number = 1e-9
        if eps.item() < 1e-5:
            warnings.warn("Too small eps: {eps} is provided. Consider using double precision")
    
    l = torch.arange(lmax+1, device=omg.device, dtype=torch.long)
    omg = omg[...,None]
    sum = (2*l+1)    *    torch.exp(-l*(l+1) * eps)    *    (  torch.sin((l+0.5)*omg) + (l+0.5)*small_number  )    /    (  torch.sin(omg/2) + 0.5*small_number  )      

    return sum.sum(dim=-1)

@torch.jit.script
def isotropic_gaussian_so3(q: torch.Tensor, eps: float, lmax: Optional[int] = None) -> torch.Tensor:
    versor = q[..., 0] # cos(omg/2)
    omg = torch.acos(versor) * 2
    assert (omg <= torch.pi).all() and (omg >= 0.).all()

    return isotropic_gaussian_so3_angle(omg=omg, eps=eps, lmax=lmax)

@torch.jit.script
def isotropic_gaussian_so3_lie_deriv(q: torch.Tensor, eps: float, lmax: Optional[int] = None) -> torch.Tensor:
    versor = q[..., 0] # cos(omg/2)
    omg = torch.acos(versor) * 2
    assert (omg <= torch.pi).all() and (omg >= 0.).all()

    if lmax is None:
        if isinstance(eps, torch.Tensor):
            lmax = determine_lmax(eps=eps.item())
        else:
            lmax = determine_lmax(eps=eps)
        
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, device=omg.device, dtype=omg.dtype)

    if eps.dtype is torch.float64:
        small_number = 1e-20
        if eps.item() < 1e-10:
            warnings.warn("Too small eps: {eps} is provided.")
    else:
        small_number = 1e-9
        if eps.item() < 1e-5:
            warnings.warn("Too small eps: {eps} is provided. Consider using double precision")
    
    l = torch.arange(lmax+1, device=q.device, dtype=torch.long) # shape: (lmax+1,)
    omg = omg[...,None] # shape: (..., 1)

    lie_deriv_cos_omg = -2 * versor[...,None] * q[...,1:] # shape: (..., 3)

    char_deriv = (((l+1) * torch.sin((l)*omg)) - ((l) * torch.sin((l+1)*omg)) + small_number*l*(l+1)*(2*l+1)) / ((1-torch.cos(omg))*torch.sin(omg) + 3*small_number) # shape: (..., lmax_+1)
    sum = ((2*l+1)    *    torch.exp(-l*(l+1) * eps)    *    char_deriv).unsqueeze(-1) * lie_deriv_cos_omg.unsqueeze(-2)   # shape: (..., lmax_+1, 3)

    return sum.sum(dim=-2)

@torch.jit.script
def isotropic_gaussian_so3_score(q: torch.Tensor, eps: float, lmax: Optional[int] = None) -> torch.Tensor:
    deriv = isotropic_gaussian_so3_lie_deriv(q=q, eps=eps, lmax=lmax)
    prob = isotropic_gaussian_so3(q=q, eps=eps, lmax=lmax)

    if q.dtype is torch.float64:
        small_number = 1e-30
    else:
        small_number = 1e-10

    return deriv / (prob.unsqueeze(-1) + small_number)



class IgSO3Dist(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_symmetric = True
        self.small_eps_criteria = 0.05
    
    def isotropic_gaussian_so3_angle(self, omg: Union[float, torch.Tensor], eps: float, lmax = None) -> torch.Tensor:
        # if eps <= 1.:
        #     assert lmax is None
        #     return isotropic_gaussian_so3_small(omg, eps)
        # else:
        #     return isotropic_gaussian_so3(omg=omg, eps=eps, lmax=lmax)
        return isotropic_gaussian_so3_angle(omg=omg, eps=eps, lmax=lmax)

    def _get_inv_cdf(self, eps: float, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None):
        X = torch.linspace(0, math.pi, 300, device=device, dtype=dtype)
        Y = self.isotropic_gaussian_so3_angle(X, eps=eps) * haar_measure_angle(X)

        cdf = torch.cumsum(Y, dim=-1)
        cdf = cdf / cdf.max()
        return Interp1D(cdf, X, 'linear') # https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1
    
    def _sample(self, N, eps, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        inverse_cdf = self._get_inv_cdf(eps=eps, device=device, dtype=dtype)
        angle = inverse_cdf(torch.rand(N, device=device, dtype=dtype)).unsqueeze(-1)
        axis = F.normalize(torch.randn(N,3, device=device, dtype=dtype), dim=-1)

        return transforms.axis_angle_to_quaternion(axis * angle)
    
    def _sample_approx(self, N, eps, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        return transforms.axis_angle_to_quaternion(torch.randn(N,3, device=device, dtype=dtype) * math.sqrt(2*eps))

    def sample(self, eps: float, N: int = 1, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        if eps <= self.small_eps_criteria:
            return self._sample_approx(N=N, eps=eps, device=device, dtype=dtype)
        else:
            return self._sample(N=N, eps=eps, device=device, dtype=dtype)
    
    def log_likelihood(self, q: torch.Tensor, eps: float):
        angle: torch.Tensor = torch.norm(transforms.quaternion_to_axis_angle(transforms.standardize_quaternion(q)), dim=-1)
        if eps <= self.small_eps_criteria:
            logP= -0.25 * angle.square() / eps - 1.5*math.log(4*eps*math.pi) # gaussian
            logP = logP + torch.log(4*math.pi*angle.square()) # d^3x = r^2 sin(thete) dr d(theta) d(phi) => d^3x = {omg^2 d(omg)} x {d(SolidAngle)} = {4*pi*omg^2 d(omg)} x {d(NormalizedSolidAngle)}, where \int d(SolidAngle) = 4pi => d(SolidAngle) = 4pi * d(NormalizedSolidAngle)
            logP = logP - torch.log(haar_measure_angle(angle))
            return logP
        else:
            logP = torch.log(self.isotropic_gaussian_so3_angle(angle, eps=eps))
            return logP














class GaussianDistR3(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        self.register_buffer('dummy', torch.tensor([0]), persistent=False)
        self.is_symmetric = True
        self.is_isotropic = True

    def sample(self, N=1):
        return self.std * torch.randn(N,3, device=self.dummy.device)

    def propose(self, X):
        X_new = X.reshape(-1,3)
        N = len(X_new)
        return (X_new + self.sample(N)).reshape(*(X.shape))

    def log_gaussian_factor(self, X):
        return -0.5/(self.std**2) * (X**2).sum(-1)

    def log_factor(self, X2, X1 = None): # log Q(x2 | x1)
        if X1 is None:
            X_diff = X2
        else:
            assert X1.shape == X2.shape
            X_diff = X2 - X1

        return self.log_gaussian_factor(X_diff)

    def log_factor_diff(self, X2, X1): # log[ Q(x1 | x2) / Q(x2 | x1) ]
        assert X2.shape == X1.shape
        return torch.zeros(X2.shape[:-1], device=X2.device)











class UniformDistR3(nn.Module):
    def __init__(self, ranges):
        super().__init__()
        self.register_buffer('ranges', ranges, persistent=False) # (3,2)
        self.is_symmetric = True
        self.is_isotropic = False

    def sample(self, N=1):
        return torch.rand(N,3, device=self.ranges.device) * (self.ranges[:,1] - self.ranges[:,0]) + self.ranges[:,0]

    def propose(self, X):
        N = len(X)
        X_new = self.sample(N)
        return X_new.reshape(*(X.shape))

    def log_factor(self, X2, X1): #Q(X2|X1)
        assert X2.shape == X1.shape
        in_range = (self.ranges[:,1] >= X2) * (X2 >= self.ranges[:,0])
        
        return (~in_range).any(dim=-1) * -30

    def log_factor_diff(self, X2, X1): # log[ Q(x1 | x2) / Q(x2 | x1) ]
        assert X2.shape == X1.shape
        return torch.zeros(X2.shape[:-1], device=X2.device)











class DistSE3(nn.Module):
    def __init__(self, decoupled = False):
        super().__init__()
        self.dist_X = None
        self.dist_R = None
        self.decoupled = decoupled

    def sample(self, N=1):
        q = self.dist_R.sample(N=N)
        x = self.dist_X.sample(N=N)
        x = transforms.quaternion_apply(q, x)    # TODO: this does not work well with UniformDist so needs to be fixed
        return torch.cat([q, x], dim=-1)

    def propose(self, T):                                               
        q_old, X_old = T[...,:4], T[...,4:]                             
        q_new = self.dist_R.propose(q_old)                              
        X_prop = self.dist_X.propose(torch.zeros_like(X_old))           
        if self.decoupled:
            X_new = X_old + X_prop
        else:
            X_new = transforms.quaternion_apply(q_old, X_prop) + X_old
        
        return torch.cat([q_new, X_new], dim=-1)

    def log_factor(self, T2, T1):                                                                                   
        q2, X2 = T2[...,:4], T2[...,4:]                      
        q1, X1 = T1[...,:4], T1[...,4:]                                       
        
        log_factor_q = self.dist_R.log_factor(q2, q1)
        if self.decoupled:
            X_prop = X2-X1
        else:
            X_prop = transforms.quaternion_apply(transforms.quaternion_invert(q1), X2-X1) # X2 and X1 are represented in space frame, but DeltaX is sampled from gaussian in (old) body frame(=q1) so X2-X1 should be transported to q1 body frame
        log_factor_X = self.dist_X.log_factor(X_prop, torch.zeros_like(X_prop))

        return log_factor_q + log_factor_X

    def log_factor_diff(self, T2, T1): # log Q(T1 | T2) / logQ(T2 | T1)                   # (Note that numerator is T1 | T2, not T2 | T1 since we're doing MCMC)
        q2, X2 = T2[...,:4], T2[...,4:]
        q1, X1 = T1[...,:4], T1[...,4:]
        
        log_factor_q_diff = self.dist_R.log_factor_diff(q2, q1)

        if self.dist_X.is_isotropic is True or self.decoupled:
            log_factor_X_diff = self.dist_X.log_factor_diff(X2, X1)
        else:
            X_prop_21 = transforms.quaternion_apply(transforms.quaternion_invert(q1), X2-X1)
            X_prop_12 = transforms.quaternion_apply(transforms.quaternion_invert(q2), X1-X2)
            log_factor_X_21 = self.dist_X.log_factor(X_prop_21, torch.zeros_like(X_prop_21))
            log_factor_X_12 = self.dist_X.log_factor(X_prop_12, torch.zeros_like(X_prop_12))
            log_factor_X_diff = log_factor_X_12 - log_factor_X_21

        return log_factor_q_diff + log_factor_X_diff







class GaussianDistSE3(DistSE3):
    def __init__(self, std_theta, std_X, decoupled = False):
        super().__init__(decoupled=decoupled)
        self.dist_R = IgSO3Dist(std=std_theta)
        self.dist_X = GaussianDistR3(std=std_X)


class UniformDistSE3(DistSE3):
    def __init__(self, ranges_X, decoupled = False):
        super().__init__(decoupled=decoupled)
        self.dist_R = QuaternionUniformDist()
        self.dist_X = UniformDistR3(ranges = ranges_X)



