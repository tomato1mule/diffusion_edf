import warnings
from typing import List, Tuple, Dict, Optional, Union, Callable
import itertools
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_cluster import radius_graph, radius
from torch_scatter import scatter, scatter_mean

from e3nn import o3
import e3nn.nn

from diffusion_edf.so3_util import IrrepwiseApplyScalar, IrrepwiseDotProduct


class EquivariantLayerNorm(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, affine: bool = True, trainable: bool = True, eps:float = 1e-5):
        super().__init__()
        self.affine = affine
        self.trainable = trainable
        self.irreps = o3.Irreps(irreps)
        self.dim = self.irreps.dim
        self.scalar_dim = 0
        for (n, (l,p)) in self.irreps:
            if p != 1:
                raise NotImplementedError("Only tested for SE(3)-equivariance. E(3) is not considered yet.")
            if l==0:
                self.scalar_dim += n
            if n==0:
                raise NotImplementedError(f"n=0 is not supported")
            elif n==1:
                warnings.warn(f"Irreps with n=1 is provided to layernorm. (Irreps: {self.irreps}")

        self.dp = IrrepwiseDotProduct(irreps=self.irreps)
        self.irrepwise_mul = IrrepwiseApplyScalar(irreps=self.irreps, binary_ops=torch.mul)
        self.beta = 1.
        self.softplus = torch.nn.Softplus(beta=1.)
        weight = torch.ones(self.dim) / self.beta * math.log(math.exp(self.beta)-1) # Shape: (..., D)

        if self.affine:
            bias = torch.zeros(self.scalar_dim)
        else:
            bias = None

        if self.trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = None if bias is None else torch.nn.Parameter(bias)
        else:
            self.register_buffer("weight", weight)
            if bias is None:
                self.bias = None
            else:
                self.register_buffer("bias", bias)

        self.eps = eps


        scalar_indices = torch.empty(0, dtype=torch.long)
        scalar_group = torch.empty(0, dtype=torch.long)
        self.n_scalar_group = 0
        for (n,(l,p)), slice in zip(self.irreps, self.irreps.slices()):
            if l != 0:
                continue
            else:
                start, stop, step = slice.start, slice.stop, slice.step
                if step is None:
                    step = 1
                else:
                    assert isinstance(step, int)
                indices = torch.arange(start=start, end=stop, step=step, dtype=scalar_indices.dtype)
                groups = torch.ones_like(indices) * self.n_scalar_group
                scalar_indices = torch.cat([scalar_indices, indices])
                scalar_group = torch.cat([scalar_group, groups])
                self.n_scalar_group += 1
        

        self.register_buffer("scalar_indices", scalar_indices) # Shape: (nScalar, )
        self.register_buffer("scalar_group", scalar_group) # Shape: (nScalar, )
        assert self.scalar_indices.shape[-1] == self.scalar_group.shape[-1] == self.scalar_dim, f"scalar_indices: {self.scalar_indices.shape}, scalar_group: {self.scalar_group.shape}"

    def aggregate_scalar(self, x: torch.Tensor) -> torch.Tensor: # (..., D) -> (..., nScalar)
        assert x.shape[-1] == self.dim
        return torch.index_select(input=x, dim=-1, index=self.scalar_indices)
    
    def scatter_scalar(self, x: torch.Tensor) -> torch.Tensor:   # (..., nScalar) -> (..., D)
        assert x.shape[-1] == self.scalar_dim
        return scatter(src=x, index=self.scalar_indices, dim=-1, dim_size=self.dim)
    
    def get_bias(self) -> torch.Tensor:
        assert self.bias is not None
        return self.scatter_scalar(self.bias) # Shape: (D, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dim    # x: Shape: (..., D)

        scalar = self.aggregate_scalar(x) # Shape: (..., nScalar)
        scalar_mean = scatter_mean(src=scalar, dim=-1, index=self.scalar_group) # Shape: (..., nGroup)
        scalar_mean = torch.index_select(input=scalar_mean, dim=-1, index=self.scalar_group) # Shape: (..., nScalar)
        scalar_mean = self.scatter_scalar(scalar_mean) # Shape: (..., D)
        x = x - scalar_mean # Shape: (..., D)

        rnorm = torch.rsqrt(self.dp(x,x)+self.eps) # Shape: (..., nIrrep)
        x = self.irrepwise_mul(x, rnorm) * self.softplus(self.weight) # Shape: (..., D)
        if self.bias is not None:
            x = x + self.get_bias()

        return x

    def extra_repr(self) -> str:
        return 'irreps={irreps}, eps={eps}, affine={affine}, trainable={trainable}'.format(**self.__dict__)

    @staticmethod
    def test() -> bool:
        irreps_list = [o3.Irreps("3x1e+10x0e+5x2e"), o3.Irreps("3x0e+10x0e+5x1e+4x0e"), o3.Irreps('1x0e'), o3.Irreps('1x0e'), o3.Irreps('3x1e'), o3.Irreps('3x1e')]
        jits = ['default', 'jit']

        for context in itertools.product(irreps_list, jits):
            irreps, jit = context
            module = EquivariantLayerNorm(irreps=irreps, eps=1e-8)
            if jit == 'default':
                pass
            elif jit == 'jit':
                module = torch.jit.script(module)
            else:
                raise ValueError(f"Unknown jit type {jit}")
            device = module.weight.device
            dtype = module.weight.dtype

            x = irreps.randn(5,4,3,2,-1, device = device, dtype = dtype)
            z = module(x).detach()

            for (n,(l,p)), slice in zip(irreps, irreps.slices()):
                mean = z[...,slice].mean(dim=-1)
                norm = torch.norm(z[...,slice], dim=-1)
                if l==0:
                    result = torch.isclose(mean, torch.zeros_like(mean), atol=1e-3).type(torch.double).mean()
                    if result.item() < 0.9999:
                        warnings.warn(f"EquivariantLayernorm mean=0 test failed for irreps {irreps} || jit type: {jit} || isclose ratio: {result.item()}")
                        return False
                if n>=2:
                    result = torch.isclose(norm, torch.ones_like(norm), atol=1e-3).type(torch.double).mean()
                    if result.item() < 0.9999:
                        warnings.warn(f"EquivariantLayernorm norm=1 test failed for irreps {irreps} || jit type: {jit} || isclose ratio: {result.item()}")
                        return False

        return True