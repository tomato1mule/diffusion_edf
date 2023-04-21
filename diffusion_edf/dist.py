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
from diffusion_edf import SE3_SCORE_TYPE
from diffusion_edf import transforms
from xitorch.interpolate import Interp1D



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
def igso3_small_angle(omg: torch.Tensor, eps: Union[float, torch.Tensor]) -> torch.Tensor:
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
def igso3_angle(omg: torch.Tensor, eps: Union[float, torch.Tensor], lmax: Optional[int] = None) -> torch.Tensor:
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

    return torch.clamp(sum.sum(dim=-1), min = 0.)

@torch.jit.script
def igso3(q: torch.Tensor, eps: Union[float, torch.Tensor], lmax: Optional[int] = None) -> torch.Tensor:
    versor = q[..., 0] # cos(omg/2)
    omg = torch.acos(versor) * 2
    assert (omg <= torch.pi).all() and (omg >= 0.).all()

    return igso3_angle(omg=omg, eps=eps, lmax=lmax)

@torch.jit.script
def igso3_lie_deriv(q: torch.Tensor, eps: Union[float, torch.Tensor], lmax: Optional[int] = None) -> torch.Tensor:
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
def igso3_score(q: torch.Tensor, eps: Union[float, torch.Tensor], lmax: Optional[int] = None) -> torch.Tensor:
    deriv = igso3_lie_deriv(q=q, eps=eps, lmax=lmax)
    prob = igso3(q=q, eps=eps, lmax=lmax).unsqueeze(-1)

    if q.dtype is torch.float64:
        small_number = 1e-30
    else:
        small_number = 1e-10

    return (deriv / (prob + small_number)) * (prob > 0.)


def get_inv_cdf(eps: Union[float, torch.Tensor], 
                N:int = 1000, 
                dtype: Optional[torch.dtype] = torch.float64, 
                device: Optional[Union[str, torch.device]] = None) -> Interp1D:
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, device=device, dtype=dtype)

    N=1000
    omg_max_prob = 2*math.sqrt(eps)
    omg_range = min(omg_max_prob * 4, math.pi)
    # omg_max_prob_idx = ((omg_max_prob) * N / omg_range)

    X = torch.linspace(0, omg_range, N, device=device, dtype=dtype)
    Y = igso3_angle(X, eps=eps) * haar_measure_angle(X)

    cdf = torch.cumsum(Y, dim=-1)
    cdf = cdf / cdf.max()
    return Interp1D(cdf, X, 'linear') # https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1

def _sample_igso3(inv_cdf: Interp1D, 
                  N: int, 
                  dtype: Optional[torch.dtype] = torch.float64, 
                  device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    angle = inv_cdf(torch.rand(N, device=device, dtype=dtype)).unsqueeze(-1)
    axis = F.normalize(torch.randn(N,3, device=device, dtype=dtype), dim=-1)

    return transforms.axis_angle_to_quaternion(axis * angle)

def sample_igso3(eps: Union[float, torch.Tensor], 
                 N: int = 1, 
                 dtype: Optional[torch.dtype] = torch.float64, 
                 device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    inv_cdf = get_inv_cdf(eps=eps, device=device, dtype=dtype)
    return _sample_igso3(inv_cdf=inv_cdf, N=N, device=device, dtype=dtype)

@torch.jit.script
def r3_isotropic_gaussian_score(x: torch.Tensor, std: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=x.device, dtype=x.dtype)
    return -x / torch.square(std)

@torch.jit.script
def r3_log_isotropic_gaussian(x: torch.Tensor, std: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=x.device, dtype=x.dtype)
    
    return -0.5 * torch.square(x).sum(dim=-1) / torch.square(std) - 1.5*math.log(2*torch.square(std)*torch.pi) # gaussian

@torch.jit.script
def r3_isotropic_gaussian(x: torch.Tensor, std: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=x.device, dtype=x.dtype)
    
    return torch.exp(r3_log_isotropic_gaussian(x=x, std=std)) # gaussian

@torch.jit.script
def se3_isotropic_gaussian_score(T: torch.Tensor, 
                                 eps: Union[float, torch.Tensor], 
                                 std: Union[float, torch.Tensor]) -> SE3_SCORE_TYPE:
    q = T[..., :4]
    x = T[..., 4:]

    ang_score = igso3_score(q=q, eps=eps)
    lin_score = r3_isotropic_gaussian_score(x=x, std=std)
    lin_score = transforms.quaternion_apply(transforms.quaternion_invert(q), lin_score)
    
    return ang_score, lin_score

@torch.jit.script
def adjoint_se3_score(T_ref: torch.Tensor, ang_score: torch.Tensor, lin_score: torch.Tensor) -> SE3_SCORE_TYPE:
    assert ang_score.shape[:-1] == lin_score.shape[:-1] == T_ref.shape[:-1]
    assert T_ref.shape[-1] == 7
    
    ang_score = transforms.quaternion_apply(T_ref[..., :4], ang_score)
    lin_score = torch.cross(T_ref[...,4:], ang_score, dim=-1) + transforms.quaternion_apply(T_ref[..., :4], lin_score)

    return ang_score, lin_score

@torch.jit.script
def adjoint_isotropic_se3_score(x_ref: torch.Tensor, ang_score: torch.Tensor, lin_score: torch.Tensor) -> SE3_SCORE_TYPE:
    assert ang_score.shape[:-1] == lin_score.shape[:-1] == x_ref.shape[:-1]
    assert x_ref.shape[-1] == 3
    
    lin_score = torch.cross(x_ref, ang_score, dim=-1) + lin_score

    return ang_score, lin_score

@torch.jit.script
def adjoint_inv_tr_se3_score(T_ref: torch.Tensor, ang_score: torch.Tensor, lin_score: torch.Tensor) -> SE3_SCORE_TYPE:
    assert ang_score.shape[:-1] == lin_score.shape[:-1] == T_ref.shape[:-1]
    assert T_ref.shape[-1] == 7
    
    lin_score = transforms.quaternion_apply(T_ref[..., :4], lin_score)
    ang_score = transforms.quaternion_apply(T_ref[..., :4], ang_score) + torch.cross(T_ref[...,4:], lin_score, dim=-1)

    return ang_score, lin_score

@torch.jit.script
def adjoint_inv_tr_isotropic_se3_score(x_ref: torch.Tensor, ang_score: torch.Tensor, lin_score: torch.Tensor) -> SE3_SCORE_TYPE:
    assert ang_score.shape[:-1] == lin_score.shape[:-1] == x_ref.shape[:-1]
    assert x_ref.shape[-1] == 3
    
    ang_score = ang_score + torch.cross(x_ref, lin_score, dim=-1)

    return ang_score, lin_score

def sample_isotropic_se3_gaussian(eps: Union[float, torch.Tensor], std: Union[float, torch.Tensor], N: int = 1, dtype: Optional[torch.dtype] = torch.float64, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    x = torch.randn(N, 3, device=device, dtype=dtype) * std
    q = sample_igso3(eps=eps, N=N, dtype=dtype, device=device)
    return torch.cat([q,x], dim=-1)


def diffuse_isotropic_se3(T0: torch.Tensor, 
                          eps: Union[float, torch.Tensor], 
                          std: Union[float, torch.Tensor], 
                          x_ref: Optional[torch.Tensor] = None, 
                          double_precision: bool = True) -> Tuple[torch.Tensor, 
                                                                  torch.Tensor, 
                                                                  SE3_SCORE_TYPE, 
                                                                  SE3_SCORE_TYPE]:
    assert T0.ndim == 2 and T0.shape[-1] == 7  # T0: shape (nT, 7)
    assert x_ref.ndim == 2 and x_ref.shape[-1] == 3 # x_ref: shape (nT, 3)

    input_dtype = T0.dtype
    if double_precision:
        T0 = T0.type(dtype=torch.float64)
        if isinstance(eps, torch.Tensor):
            eps = eps.type(dtype=torch.float64)
        if isinstance(std, torch.Tensor):
            std = std.type(dtype=torch.float64)
        if isinstance(x_ref, torch.Tensor):
            x_ref = x_ref.type(dtype=torch.float64) 

    delta_T = sample_isotropic_se3_gaussian(eps=eps, std=std, N=len(T0), dtype=T0.dtype, device=T0.device)     # shape: (nT, 7)
    ang_score_ref, lin_score_ref = se3_isotropic_gaussian_score(T=delta_T, eps=eps, std=std)                   # shape: (nT, 3), (nT, 3)
    if x_ref is not None:
        ang_score, lin_score = adjoint_inv_tr_isotropic_se3_score(x_ref=x_ref, ang_score=ang_score_ref, lin_score=lin_score_ref)  # shape: (nT, 3), (nT, 3)
    else:
        ang_score, lin_score = ang_score_ref, lin_score_ref    # shape: (nT, 3), (nT, 3)

    if x_ref is not None:
        delta_T = torch.cat([delta_T[...,:4],
                             delta_T[...,4:] + x_ref - transforms.quaternion_apply(delta_T[...,:4], x_ref)
                            ], dim=-1)        # shape: (nT, 7)

    T = transforms.multiply_se3(T0, delta_T) # shape: (nT, 7)

    return (
        T.type(dtype=input_dtype), 
        delta_T.type(dtype=input_dtype), 
        (ang_score.type(dtype=input_dtype), lin_score.type(dtype=input_dtype)), 
        (ang_score_ref.type(dtype=input_dtype), lin_score_ref.type(dtype=input_dtype))
    )


def diffuse_isotropic_se3_batched(T0: torch.Tensor, 
                          eps: Union[float, torch.Tensor], 
                          std: Union[float, torch.Tensor], 
                          x_ref: Optional[torch.Tensor], 
                          double_precision: bool = True) -> Tuple[torch.Tensor, 
                                                                  torch.Tensor, 
                                                                  SE3_SCORE_TYPE, 
                                                                  SE3_SCORE_TYPE]:
    assert T0.ndim == 2 and T0.shape[-1] == 7  # T0: shape (nT, 7)

    if x_ref is not None:
        assert x_ref.ndim == 2 and x_ref.shape[-1] == 3 # x_ref: shape (nT, 3)

    input_dtype = T0.dtype
    if double_precision:
        T0 = T0.type(dtype=torch.float64)
        if isinstance(eps, torch.Tensor):
            eps = eps.type(dtype=torch.float64)
        if isinstance(std, torch.Tensor):
            std = std.type(dtype=torch.float64)
        if isinstance(x_ref, torch.Tensor):
            x_ref = x_ref.type(dtype=torch.float64) 

    delta_T = sample_isotropic_se3_gaussian(eps=eps, std=std, N=len(x_ref) * len(T0), dtype=T0.dtype, device=T0.device)     # shape: (nXref*nT, 7)
    ang_score_ref, lin_score_ref = se3_isotropic_gaussian_score(T=delta_T, eps=eps, std=std)                   # shape: (nXref*nT, 3), (nXref*nT, 3)
    if x_ref is not None:
        ang_score, lin_score = adjoint_inv_tr_isotropic_se3_score(x_ref=x_ref, ang_score=ang_score_ref, lin_score=lin_score_ref)  # shape: (nXref*nT, 3), (nXref*nT, 3)
    else:
        ang_score, lin_score = ang_score_ref, lin_score_ref    # shape: (nXref*nT, 3), (nXref*nT, 3)

    delta_T = delta_T.view(len(x_ref),*T0.shape)               # shape: (nXref, nT, 7)
    ang_score = ang_score.view(len(x_ref),*T0.shape[:-1], 3)   # shape: (nXref, nT, 3)
    lin_score = lin_score.view(len(x_ref),*T0.shape[:-1], 3)   # shape: (nXref, nT, 3)
    ang_score_ref = ang_score_ref.view(len(x_ref),*T0.shape[:-1], 3)  # shape: (nXref, nT, 3)
    lin_score_ref = lin_score_ref.view(len(x_ref),*T0.shape[:-1], 3)  # shape: (nXref, nT, 3)

    if x_ref is not None:
        delta_T = torch.cat([delta_T[...,:4],
                             delta_T[...,4:] + x_ref.unsqueeze(-2) - transforms.quaternion_apply(delta_T[...,:4], x_ref.unsqueeze(-2))
                            ], dim=-1)        # shape: (nXref, nT, 7)

    T = transforms.multiply_se3(T0.unsqueeze(-3), delta_T) # shape: (nXref, nT, 7)

    return (
        T.type(dtype=input_dtype),                  # shape: (nXref, nT, 7)
        delta_T.type(dtype=input_dtype),            # shape: (nXref, nT, 7)
        (ang_score.type(dtype=input_dtype), lin_score.type(dtype=input_dtype)),  # shape: (nXref, nT, 3), (nXref, nT, 3),
        (ang_score_ref.type(dtype=input_dtype), lin_score_ref.type(dtype=input_dtype))   # shape: (nXref, nT, 3), (nXref, nT, 3),
    )     




















# class IgSO3Dist(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.is_symmetric = True
#         self.small_eps_criteria = 0.05
    
#     def isotropic_gaussian_so3_angle(self, omg: Union[float, torch.Tensor], eps: float, lmax = None) -> torch.Tensor:
#         # if eps <= 1.:
#         #     assert lmax is None
#         #     return isotropic_gaussian_so3_small(omg, eps)
#         # else:
#         #     return isotropic_gaussian_so3(omg=omg, eps=eps, lmax=lmax)
#         return isotropic_gaussian_so3_angle(omg=omg, eps=eps, lmax=lmax)

#     def _get_inv_cdf(self, eps: float, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None):
#         X = torch.linspace(0, math.pi, 300, device=device, dtype=dtype)
#         Y = self.isotropic_gaussian_so3_angle(X, eps=eps) * haar_measure_angle(X)

#         cdf = torch.cumsum(Y, dim=-1)
#         cdf = cdf / cdf.max()
#         return Interp1D(cdf, X, 'linear') # https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1
    
#     def _sample(self, N, eps, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
#         inverse_cdf = self._get_inv_cdf(eps=eps, device=device, dtype=dtype)
#         angle = inverse_cdf(torch.rand(N, device=device, dtype=dtype)).unsqueeze(-1)
#         axis = F.normalize(torch.randn(N,3, device=device, dtype=dtype), dim=-1)

#         return transforms.axis_angle_to_quaternion(axis * angle)
    
#     def _sample_approx(self, N, eps, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
#         return transforms.axis_angle_to_quaternion(torch.randn(N,3, device=device, dtype=dtype) * math.sqrt(2*eps))

#     def sample(self, eps: float, N: int = 1, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
#         if eps <= self.small_eps_criteria:
#             return self._sample_approx(N=N, eps=eps, device=device, dtype=dtype)
#         else:
#             return self._sample(N=N, eps=eps, device=device, dtype=dtype)
    
#     def log_likelihood(self, q: torch.Tensor, eps: float):
#         angle: torch.Tensor = torch.norm(transforms.quaternion_to_axis_angle(transforms.standardize_quaternion(q)), dim=-1)
#         if eps <= self.small_eps_criteria:
#             logP= -0.25 * angle.square() / eps - 1.5*math.log(4*eps*math.pi) # gaussian
#             logP = logP + torch.log(4*math.pi*angle.square()) # d^3x = r^2 sin(thete) dr d(theta) d(phi) => d^3x = {omg^2 d(omg)} x {d(SolidAngle)} = {4*pi*omg^2 d(omg)} x {d(NormalizedSolidAngle)}, where \int d(SolidAngle) = 4pi => d(SolidAngle) = 4pi * d(NormalizedSolidAngle)
#             logP = logP - torch.log(haar_measure_angle(angle))
#             return logP
#         else:
#             logP = torch.log(self.isotropic_gaussian_so3_angle(angle, eps=eps))
#             return logP



