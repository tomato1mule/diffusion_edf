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

from diffusion_edf import irreps_util


class EquivariantLayerNorm(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, affine: bool = True, trainable: bool = True, eps: float = 1e-5, softplus: bool = False, normalization: str = 'component'):
        super().__init__()
        self.affine = affine
        self.trainable = trainable
        self.irreps = o3.Irreps(irreps)
        if normalization not in ['component', 'norm']:
            raise ValueError(f"Unknown normalization type: {normalization}")
        self.normalization = normalization
        if not irreps_util.check_irreps_parity(irreps=irreps, parity=1):
            raise NotImplementedError("Only tested for SE(3)-equivariance. E(3) is not tested yet.")
        for (n, (l,p)) in self.irreps:
            if n==0:
                raise NotImplementedError(f"n=0 is not supported")
            elif n==1:
                warnings.warn(f"Irreps with n=1 is provided to layernorm. (Irreps: {self.irreps}")

        self.dim = self.irreps.dim
        self.n_irreps = self.irreps.num_irreps
        self.scalar_dim = self.irreps.count(ir = "0e")
        self.n_irreps_group = len(self.irreps)
        self.register_buffer("ls", torch.tensor(self.irreps.ls, dtype=torch.long), persistent=False) # Shape: (nIrreps)

        group_scatter_indices = irreps_util.get_group_scatter_indices(irreps=irreps)
        self.register_buffer("group_scatter_indices", group_scatter_indices, persistent=False)            # Shape: (D, )
        assert self.group_scatter_indices.shape[-1] == self.dim and self.group_scatter_indices.ndim == 1

        irrep_scatter_indices = irreps_util.get_irrep_scatter_indices(irreps=irreps)
        self.register_buffer("irrep_scatter_indices", irrep_scatter_indices, persistent=False)            # Shape: (D, )
        assert self.irrep_scatter_indices.shape[-1] == self.dim and self.irrep_scatter_indices.ndim == 1

        irrep_to_group_scatter_indices = irreps_util.get_irrep_to_group_scatter_indices(irreps=irreps)
        self.register_buffer("irrep_to_group_scatter_indices", irrep_to_group_scatter_indices, persistent=False)            # Shape: (nIrrep, )
        assert self.irrep_to_group_scatter_indices.shape[-1] == self.n_irreps and self.irrep_to_group_scatter_indices.ndim == 1

        self.irrepwise_mul = irreps_util.IrrepwiseApplyScalar(irreps=self.irreps, binary_ops=torch.mul)
        
        if softplus:
            self.beta = 1.
            self.softplus = torch.nn.Softplus(beta=1.)
            weight = torch.ones(self.n_irreps) / self.beta * math.log(math.exp(self.beta)-1) # Shape: (nIrreps, )
        else:
            self.beta = None
            self.softplus = None
            weight = torch.ones(self.n_irreps)   # Shape: (nIrreps, )

        assert weight.shape[-1] == self.n_irreps and weight.ndim == 1
        

        if self.affine:
            bias = torch.zeros(self.scalar_dim)      # Shape: (nScalar, )
        else:
            bias = None

        if self.trainable:
            self.weight = torch.nn.Parameter(weight) # Shape: (nIrreps, )
            self.bias = None if bias is None else torch.nn.Parameter(bias)
        else:
            self.register_buffer("weight", weight) # Shape: (nIrreps, )
            if bias is None:
                self.bias = None
            else:
                self.register_buffer("bias", bias) # Shape: (nScalar, )

        self.eps = eps


        scalar_indices = torch.empty(0, dtype=torch.long)
        scalar_group = torch.empty(0, dtype=torch.long)
        self.n_scalar_group = 0
        for (n,(l,p)), slice in zip(self.irreps, self.irreps.slices()):
            if l != 0:
                continue
            else:
                start, stop = slice.start, slice.stop
                indices = torch.arange(start=start, end=stop, dtype=scalar_indices.dtype)
                groups = torch.ones_like(indices) * self.n_scalar_group
                scalar_indices = torch.cat([scalar_indices, indices])
                scalar_group = torch.cat([scalar_group, groups])
                self.n_scalar_group += 1
        

        self.register_buffer("scalar_indices", scalar_indices) # Shape: (nScalar, )
        self.register_buffer("scalar_group", scalar_group) # Shape: (nScalar, )
        assert self.scalar_indices.shape[-1] == self.scalar_group.shape[-1] == self.scalar_dim, f"scalar_indices: {self.scalar_indices.shape}, scalar_group: {self.scalar_group.shape}"
        assert self.scalar_indices.ndim==self.scalar_group.ndim==1, f"scalar_indices: {self.scalar_indices.shape}, scalar_group: {self.scalar_group.shape}"

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
        scalar_mean = scatter_mean(src=scalar, dim=-1, index=self.scalar_group) # Shape: (..., nScalarGroup)
        scalar_mean = torch.index_select(input=scalar_mean, dim=-1, index=self.scalar_group) # Shape: (..., nScalar)
        scalar_mean = self.scatter_scalar(scalar_mean) # Shape: (..., D)
        x = x - scalar_mean # Shape: (..., D)

        var = x.square() # Shape: (..., D)
        if self.normalization == 'norm':
            var = scatter(src=var, index=self.irrep_scatter_indices, dim=-1) # Shape: (..., nIrrep)
            var = scatter_mean(src=var, index=self.irrep_to_group_scatter_indices, dim=-1) # Shape: (..., nGroup)
        elif self.normalization == 'component':
            var = scatter_mean(src=var, index=self.group_scatter_indices, dim=-1) # Shape: (..., nGroup)

        rnorm = torch.rsqrt(var + self.eps) # Shape: (..., nGroup)
        rnorm = torch.index_select(input=rnorm, dim=-1, index=self.group_scatter_indices) # Shape (..., D)
        weight = self.weight # Shape (..., nIrreps)
        if self.softplus is not None:
            weight = self.softplus(weight) # Shape (..., nIrreps)
        rnorm = self.irrepwise_mul(rnorm, weight) # Shape (..., D)
        x = x*rnorm # Shape (..., D)
        
        if self.bias is not None:
            x = x + self.get_bias()

        return x

    def extra_repr(self) -> str:
        return 'irreps={irreps}, eps={eps}, affine={affine}, trainable={trainable}'.format(**self.__dict__)

    @staticmethod
    def test() -> bool:
        irreps_list = [o3.Irreps("3x1e+10x0e+5x2e"), o3.Irreps("3x0e+10x0e+5x1e+4x0e"), o3.Irreps('1x0e'), o3.Irreps('1x0e'), o3.Irreps('3x1e'), o3.Irreps('3x1e')]
        jits = ['default', 'jit']
        normalizations = ['component', 'norm']

        for context in itertools.product(irreps_list, jits, normalizations):
            irreps, jit, normalization = context
            module = EquivariantLayerNorm(irreps=irreps, eps=1e-8, normalization=normalization)
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
                if normalization == 'component':
                    norm = z[..., slice].square().mean(dim=-1)
                elif normalization == 'norm':
                    norm = z[..., slice].reshape(*z.shape[:-1],n,2*l+1).square().sum(dim=-1).mean(dim=-1)
                else:
                    raise ValueError(normalization)
                
                if l==0:
                    result = torch.isclose(mean, torch.zeros_like(mean), atol=1e-3).type(torch.double).mean()
                    if result.item() < 0.9999:
                        warnings.warn(f"EquivariantLayernorm mean=0 test failed for irreps {irreps} || jit type: {jit} || norm_type: {normalization} || isclose ratio: {result.item()}")
                        return False
                if n>=2:
                    result = torch.isclose(norm, torch.ones_like(norm), atol=1e-3).type(torch.double).mean()
                    if result.item() < 0.9999:
                        warnings.warn(f"EquivariantLayernorm norm=1 test failed for irreps {irreps} || jit type: {jit} || norm_type: {normalization} || isclose ratio: {result.item()}")
                        return False

            # Equivariance test        
            D = irreps.D_from_angles(*o3.rand_angles())
            z_rot = module(torch.einsum('ij,...j->...i',D,x))
            rot_z = torch.einsum('ij,...j->...i',D,z)
            result = torch.isclose(z_rot, rot_z, atol=1e-3).type(torch.double).mean()
            if result.item() < 0.9999:
                warnings.warn(f"EquivariantLayernorm equivariance test failed for irreps {irreps} || jit type: {jit} || norm_type: {normalization} || isclose ratio: {result.item()}")
                return False


        return True