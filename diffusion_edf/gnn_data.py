from typing import Tuple, List, Union, Optional, NamedTuple
from beartype import beartype

import torch
from e3nn import o3

from diffusion_edf.data import PointCloud
from diffusion_edf.wigner import TransformFeatureQuaternion
from diffusion_edf.pc_utils import transform_points

class FeaturedPoints(NamedTuple):
    x: torch.Tensor # Position
    f: torch.Tensor # Feature
    b: torch.Tensor # Batch idx
    w: Optional[torch.Tensor] = None # some optional scalar weight of the points

    # def __repr__(self) -> str:
    #     if self.w is None:
    #         return f"x: {self.x.shape} | f: {self.f.shape} | b: {self.b.shape} | w: None"
    #     else:
    #         return f"x: {self.x.shape} | f: {self.f.shape} | b: {self.b.shape} | w: {self.w.shape}"

def _featured_points_repr(p: FeaturedPoints) -> str:
    if p.w is None:
        return f"x: {p.x.shape} | f: {p.f.shape} | b: {p.b.shape} | w: None"
    else:
        return f"x: {p.x.shape} | f: {p.f.shape} | b: {p.b.shape} | w: {p.w.shape}"

@torch.jit.script
def set_featured_points_attribute(points: FeaturedPoints, x: Optional[torch.Tensor] = None, f: Optional[torch.Tensor] = None, b: Optional[torch.Tensor] = None, w: Union[str,Optional[torch.Tensor]] = '') -> FeaturedPoints:
    if x is None:
        x = points.x
    if f is None:
        f = points.f
    if b is None:
        b = points.b
    if isinstance(w, str):
        assert w == ''
        w = points.w
    return FeaturedPoints(x=x, f=f, b=b, w=w)

@torch.jit.script
def detach_featured_points(points: FeaturedPoints) -> FeaturedPoints:
    w = points.w
    if isinstance(w, torch.Tensor):
        w = w.detach()
    return FeaturedPoints(x=points.x.detach(), f=points.f.detach(), b=points.b.detach(), w=w)


def _list_merge_featured_points(pcds: List[FeaturedPoints]) -> FeaturedPoints:
    x: torch.Tensor = torch.cat([pcd.x for pcd in pcds], dim=0)
    f: torch.Tensor = torch.cat([pcd.f for pcd in pcds], dim=0)
    b: torch.Tensor = torch.cat([pcd.b for pcd in pcds], dim=0)
    for pcd in pcds:
        if pcd.w is not None:
            raise NotImplementedError
    return FeaturedPoints(x=x, f=f, b=b)

def _tuple_merge_featured_points(pcds: Tuple[FeaturedPoints]) -> FeaturedPoints:
    x: torch.Tensor = torch.cat([pcd.x for pcd in pcds], dim=0)
    f: torch.Tensor = torch.cat([pcd.f for pcd in pcds], dim=0)
    b: torch.Tensor = torch.cat([pcd.b for pcd in pcds], dim=0)
    for pcd in pcds:
        if pcd.w is not None:
            raise NotImplementedError
    return FeaturedPoints(x=x, f=f, b=b)

def merge_featured_points(pcds: Union[List[FeaturedPoints], Tuple[FeaturedPoints]]) -> FeaturedPoints:
    if isinstance(pcds, list):
        return _list_merge_featured_points(pcds=pcds)
    if isinstance(pcds, tuple):
        return _tuple_merge_featured_points(pcds=pcds)
    else:
        raise ValueError()
    
def pcd_to_featured_points(pcd: PointCloud, batch_idx: int = 0) -> FeaturedPoints:
    return FeaturedPoints(x=pcd.points, f=pcd.colors, b = torch.empty_like(pcd.points[..., 0], dtype=torch.long).fill_(batch_idx))

class TransformPcd(torch.nn.Module):
    @beartype
    def __init__(self, irreps: Optional[Union[str, o3.Irreps]]):
        super().__init__()
        if irreps is None:
            self.transform_features = None
        else:
            self.transform_features = TransformFeatureQuaternion(irreps=o3.Irreps(irreps))
            
    @torch.jit.ignore()
    def to(self, *args, **kwargs):
        for module in self.children():
            if isinstance(module, torch.nn.Module):
                module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, pcd: FeaturedPoints, Ts: torch.Tensor) -> FeaturedPoints: 
        assert Ts.ndim == 2 and Ts.shape[-1] == 7, f"{Ts.shape}" # Ts: (nT, 4+3: quaternion + position) 
        if self.transform_features is not None:
            f_transformed = self.transform_features(feature=pcd.f,  q=Ts[..., :4]) # (Nt, Np, F)
        else:
            f_transformed = pcd.f.expand(len(Ts), -1, -1)                      # (Nt, Np, F)
        x_transformed = transform_points(points=pcd.x, Ts=Ts)                  # (Nt, Np, 3)
        b_transformed = pcd.b.expand(len(Ts), -1)                              # (Nt, Np)
        w_transformed = pcd.w
        if isinstance(w_transformed, torch.Tensor):
            w_transformed = w_transformed.expand(len(Ts), -1)                  # (Nt, Np)

        return FeaturedPoints(f=f_transformed, x=x_transformed, b = b_transformed, w=w_transformed)
    
@torch.jit.script
def flatten_featured_points(points: FeaturedPoints):
    x = points.x.reshape(-1,3)
    f = points.f.reshape(-1,points.f.shape[-1])
    b = points.b.reshape(-1)
    w = points.w
    if w is not None:
        w = w.reshape(-1)
    else:
        w = None

    return FeaturedPoints(x=x, f=f, b=b, w=w)



class GraphEdge(NamedTuple):
    edge_src: torch.Tensor # Position
    edge_dst: torch.Tensor # Feature
    edge_length: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    edge_scalars: Optional[torch.Tensor] = None
    edge_log_weight: Optional[torch.Tensor] = None
    edge_log_weight_scalar: Optional[torch.Tensor] = None
    edge_log_weight_nonscalar: Optional[torch.Tensor] = None

