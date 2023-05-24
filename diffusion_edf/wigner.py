from typing import List, Dict, Union, Optional, Tuple
from beartype import beartype

import torch
from e3nn import o3
from e3nn.o3._wigner import _Jd
from e3nn.math._linalg import direct_sum
from e3nn.util.jit import compile_mode
from diffusion_edf.transforms import matrix_to_euler_angles, quaternion_to_matrix, standardize_quaternion

_Jd: Tuple[torch.Tensor] = tuple(J.detach().clone().to(dtype=torch.float32) for J in _Jd)

def quat_to_angle_fast(q: torch.Tensor) -> torch.Tensor: # >10 times faster than e3nn's quaternion_to_angle function
    ang = matrix_to_euler_angles(quaternion_to_matrix(q), "YXY").T
    return ang

def _z_rot_mat(angle: torch.Tensor, l: int) -> torch.Tensor:
    r"""
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).
    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    #shape, device, dtype = angle.shape, angle.device, angle.dtype
    # M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    assert angle.dim() == 1
    device, dtype = angle.device, angle.dtype
    M = angle.new_zeros((len(angle), 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    # M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    # M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    M[:, inds, reversed_inds] = torch.sin(frequencies * angle[:, None])
    M[:, inds, inds] = torch.cos(frequencies * angle[:, None])
    return M

def wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    r"""Wigner D matrix representation of :math:`SO(3)`.
    It satisfies the following properties:
    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`
    Code of this function has beed copied from `lie_learn <https://github.com/AMLab-Amsterdam/lie_learn>`_ made by Taco Cohen.
    Parameters
    ----------
    l : int
        :math:`l`
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.
    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    # if not l < len(_Jd):
    #     raise NotImplementedError(f'wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more')

    #alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    #J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)

    assert alpha.shape == beta.shape == gamma.shape # (Nt,)

    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc

def D_from_angles_(ls: List[int], muls: List[int], Js: List[torch.Tensor], alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> List[torch.Tensor]:
    Ds = []
    for l, mul, J in zip(ls, muls, Js):
        D = wigner_D(l, alpha, beta, gamma, J)
        for _ in range(mul):
            Ds.append(D)
    return Ds

def D_from_angles(irreps, Js, alpha, beta, gamma):
    ls = []
    muls = []
    for mul, ir in irreps:
        ls.append(ir.l)
        muls.append(mul)

    Ds = D_from_angles_(ls, muls, Js, alpha, beta, gamma)
    return direct_sum(*Ds)
    #return torch.block_diag(*Ds)

def D_from_quaternion_(ls: List[int], muls: List[int], Js: List[torch.Tensor], q: torch.Tensor) -> List[torch.Tensor]:
    angle = quat_to_angle_fast(q)
    alpha, beta, gamma = angle[0], angle[1], angle[2]
    return D_from_angles_(ls=ls, muls=muls, Js=Js, alpha=alpha, beta=beta, gamma=gamma)

def D_from_quaternion(irreps, Js, q):
    ls = []
    muls = []
    for mul, ir in irreps:
        ls.append(ir.l)
        muls.append(mul)

    Ds = D_from_quaternion_(ls, muls, Js, q)
    return direct_sum(*Ds)
    #return torch.block_diag(*Ds)


def transform_feature_slice_nonscalar(feature: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, l: int, J: torch.Tensor) -> torch.Tensor:
    assert feature.dim() == 2
    feature = feature.reshape(feature.shape[-2], -1, 2*l+1) # (N_query, mul*(2l+1)) -> (N_query, mul, 2l+1)
    D = wigner_D(l, alpha, beta, gamma, J) # (Nt, 2l+1, 2l+1)
    feature_transformed = torch.einsum('tij,qmj->tqmi', D, feature) # (Nt, 2l+1, 2l+1) x (Nq, mul, 2l+1) -> (Nt, Nq, mul, 2l+1)
    feature_transformed = feature_transformed.reshape(feature_transformed.shape[0], feature_transformed.shape[1], -1) # (Nt, N_query, mul*(2l+1))
    return feature_transformed

def transform_feature_slice(feature: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, l: int, J: torch.Tensor) -> torch.Tensor:
    assert alpha.ndim == 1
    assert feature.dim() == 2
    if l == 0:
        feature_transformed = feature.expand(len(alpha), len(feature), feature.shape[-1])
    else:
        feature_transformed = transform_feature_slice_nonscalar(feature=feature, alpha=alpha, beta=beta, gamma=gamma, l=l, J=J)
    return feature_transformed

def transform_feature_(ls: List[int], feature_slices: List[torch.Tensor], Js: List[torch.Tensor], alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    feature_transformed = []
    for l, feature_slice, J in zip(ls, feature_slices, Js):
        feature_transformed_slice = transform_feature_slice(feature_slice, alpha, beta, gamma, l, J)
        feature_transformed.append(feature_transformed_slice)
    
    return torch.cat(feature_transformed, dim=-1)

def transform_feature(irreps, feature, alpha, beta, gamma, Js):
    ls = []
    feature_slices = []
    for (mul,ir), slice_ in zip(irreps, irreps.slices()):
        ls.append(ir.l)
        feature_slices.append(feature[..., slice_])

    return transform_feature_(ls, feature_slices, Js, alpha, beta, gamma)

def transform_feature_quat_(ls: List[int], feature_slices: List[torch.Tensor], Js: List[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
    q = standardize_quaternion(q / torch.norm(q, dim=-1, keepdim=True))
    angle = quat_to_angle_fast(q)
    alpha, beta, gamma = angle[0], angle[1], angle[2]
    return transform_feature_(ls=ls, feature_slices=feature_slices, Js=Js, alpha=alpha, beta=beta, gamma=gamma)

def transform_feature_quat(irreps, feature, q, Js):
    ls = []
    feature_slices = []
    for (mul,ir), slice_ in zip(irreps, irreps.slices()):
        ls.append(ir.l)
        feature_slices.append(feature[..., slice_])

    return transform_feature_quat_(ls, feature_slices, Js, q)


class TransformFeatureQuaternion(torch.nn.Module):
    @beartype
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.ls = tuple([ir.l for mul, ir in irreps])
        self.slices = tuple([(slice_.start, slice_.stop) for slice_ in irreps.slices()])
        self.Js = tuple(_Jd[l] for l in self.ls)
        self.dim: int = o3.Irreps(irreps).dim
        
    @torch.jit.ignore()
    def to(self, *args, **kwargs):
        self.Js = tuple(_Jd[l].to(*args, **kwargs) for l in self.ls)
        for module in self.children():
            if isinstance(module, torch.nn.Module):
                module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, feature: torch.Tensor, q: torch.Tensor) -> torch.Tensor : # (N_Q, N_D) x (N_T, 4) -> (N_T, N_Q, N_D)
        assert feature.shape[-1] == self.dim
        feature_slices = []
        for slice_ in self.slices:
            feature_slices.append(feature[..., slice_[0]:slice_[1]])

        return transform_feature_quat_(ls=self.ls, feature_slices=feature_slices, Js=self.Js, q=q)



