from typing import List, Dict, Union, Optional, Tuple
from beartype import beartype

import torch
import einops

from e3nn.o3 import Irreps
from e3nn.o3._wigner import _Jd
from e3nn.math._linalg import direct_sum
from e3nn.util.jit import compile_mode
from diffusion_edf.transforms import matrix_to_euler_angles, quaternion_to_matrix, standardize_quaternion, normalize_quaternion

_Jd = [J.detach().clone() for J in _Jd]

@torch.jit.script
def quat_to_angle_fast(q: torch.Tensor) -> torch.Tensor: # >10 times faster than e3nn's quaternion_to_angle function
    ang = matrix_to_euler_angles(quaternion_to_matrix(q), "YXY").T
    return ang

@torch.jit.script
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

@torch.jit.script
def _wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    assert alpha.shape == beta.shape == gamma.shape # (Nt,)

    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc

def wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
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

    J: torch.Tensor = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    return _wigner_D(l=l, alpha=alpha, beta=beta, gamma=gamma, J=J)

@torch.jit.script
def _D_from_quaternion(l: int, q: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    assert q.ndim==2
    assert q.shape[-1] == 4
    q = normalize_quaternion(q)
    angle = quat_to_angle_fast(q)
    alpha, beta, gamma = angle[0], angle[1], angle[2]
    return _wigner_D(l=l, alpha=alpha, beta=beta, gamma=gamma, J=J)

def D_from_quaternion(l: int, q: torch.Tensor) -> torch.Tensor:
    J: torch.Tensor = _Jd[l].to(dtype=q.dtype, device=q.device)
    return _D_from_quaternion(l=l, q=q, J=J)


@torch.jit.script
def _transform_irreps(f: torch.Tensor, 
                      q: torch.Tensor, 
                      irreps: List[Tuple[int, Tuple[int, int]]], 
                      Js: List[torch.Tensor]) -> torch.Tensor: # (Np, F) x (Nt,4) -> (Nt, Np, F)
    assert f.ndim == 2
    assert q.ndim == 2 and q.shape[-1] == 4



    prev_idx: int = 0
    f_rotated: List[torch.Tensor] = []
    for i, (n, (l, p)) in enumerate(irreps):
        dim: int = (2*l+1) * n
        f_sliced = f[:, prev_idx: prev_idx + dim] # (Nt, n * (2l+1))
            
        if l != 0:
            # f_sliced = einops.rearrange(
            #     einops.einsum(
            #         D_from_quaternion(l=l, q=q, J=Js[i]),                      # (Nt, 2l+1, 2l+1)
            #         einops.rearrange(f_sliced, 'Np (n b) -> Np n b', n=n),     # (Np, n, 2l+1)
            #         'Nt a b, Np n b -> Nt Np n a'
            #     ), 'Nt Np n a -> Nt Np (n a)'                           
            # )                                                                  # (Nt, Np, n * (2l+1))
            f_sliced = torch.reshape(
                torch.einsum(
                    'tab,pnb->tpna',
                    _D_from_quaternion(l=l, q=q, J=Js[i]),                     # (Nt, 2l+1, 2l+1)
                    torch.reshape(f_sliced, (len(f), n, 2*l+1)),               # (Np, n, 2l+1)
                ), (len(q), len(f), dim)                        
            )                                                                  # (Nt, Np, n * (2l+1))
        else:
            # f_sliced = einops.repeat(f_sliced, 'Np f -> Nt Np f', Nt=len(q))   # (Nt, Np, n * (2l+1))
            f_sliced = f_sliced.expand(len(q), len(f), dim)   # (Nt, Np, n * (2l+1))
        
        f_rotated.append(f_sliced)

    return torch.cat(f_rotated, dim=-1)


def transform_irreps(f: torch.Tensor, 
                     q: torch.Tensor, 
                     irreps: Union[Irreps, List[Tuple[int, Tuple[int, int]]]]) -> torch.Tensor: # (Np, F) x (Nt,4) -> (Nt, Np, F)
    if isinstance(irreps, Irreps):
        irreps = [(n, (l,p)) for (n, (l,p)) in irreps]
    
    Js: List[torch.Tensor] = [_Jd[l].to(dtype=q.dtype, device=q.device) for (n, (l,p)) in irreps]

    return _transform_irreps(f=f, q=q, irreps=irreps, Js=Js)


class TransformFeatureQuaternion(torch.nn.Module):
    irreps: List[Tuple[int, Tuple[int, int]]]
    Js: List[torch.Tensor]

    @beartype
    def __init__(self, irreps: Irreps, device: Union[str, torch.device]):    
        super().__init__()
        self.irreps = [(n, (l,p)) for (n, (l,p)) in irreps]
        self.Js = [_Jd[l].to(dtype = torch.float32, device=device) for (n, (l,p)) in irreps]

    def forward(self, feature: torch.Tensor, q: torch.Tensor) -> torch.Tensor : # (N_Q, N_D) x (N_T, 4) -> (N_T, N_Q, N_D)
        return _transform_irreps(f=feature, q=q, irreps=self.irreps, Js=self.Js)
    
