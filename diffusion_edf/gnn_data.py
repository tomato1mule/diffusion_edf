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
    def __init__(self, irreps: Union[str, o3.Irreps], device: Union[str, torch.device]):
        super().__init__()
        self.transform_features = TransformFeatureQuaternion(irreps=o3.Irreps(irreps), device=device, compile=True)

    def forward(self, pcd: FeaturedPoints, Ts: torch.Tensor) -> FeaturedPoints: 
        assert Ts.ndim == 2
        assert Ts.shape[-1] == 7
        f_transformed = self.transform_features(feature=pcd.f,  q=Ts[..., :4]) # (Nt, Np, F)
        x_transformed = transform_points(points=pcd.x, Ts=Ts)                  # (Nt, Np, 3)
        b_transformed = pcd.b.expand(len(Ts), -1)                              # (Nt, Np)
        w_transformed = pcd.w
        if isinstance(w_transformed, torch.Tensor):
            w_transformed = w_transformed.expand(len(Ts), -1)                  # (Nt, Np)

        return FeaturedPoints(f=f_transformed, x=x_transformed, b = b_transformed, w=w_transformed)



class GraphEdge(NamedTuple):
    edge_src: torch.Tensor # Position
    edge_dst: torch.Tensor # Feature
    edge_length: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    edge_scalars: Optional[torch.Tensor] = None
    edge_log_weight: Optional[torch.Tensor] = None
    edge_log_weight_scalar: Optional[torch.Tensor] = None
    edge_log_weight_nonscalar: Optional[torch.Tensor] = None

