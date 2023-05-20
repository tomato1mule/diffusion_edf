from typing import Tuple, List, Union, Optional, NamedTuple

import torch

from diffusion_edf.data import PointCloud

# @dataclass
# class FeaturedPoints:
#     x: torch.Tensor # Position
#     f: torch.Tensor # Feature
#     b: torch.Tensor = None # Batch idx
#     s: torch.Tensor = None # Scale Space

#     def __post_init__(self):
#         assert self.x.shape[-1] == 3, f"x must be three dimensional positions, but x of shape {self.x.shape} is provided."
#         assert self.x.shape

#         if self.b is None:
#             self.b = torch.zeros_like(self.x[..., 0], dtype=torch.long)

#         if self.s is None:
#             self.s = torch.zeros_like(self.x[..., 0], dtype=torch.long)

#         assert self.x.shape[:-1] == self.f.shape[:-1] == self.b.shape == self.s.shape

# FeaturedPoints = NamedTuple('FeaturedPoints', [('x', torch.Tensor), ('f', torch.Tensor), ('b', torch.Tensor)])
class FeaturedPoints(NamedTuple):
    x: torch.Tensor # Position
    f: torch.Tensor # Feature
    b: torch.Tensor # Batch idx

def _list_merge_featured_points(pcds: List[FeaturedPoints]) -> FeaturedPoints:
    x: torch.Tensor = torch.cat([pcd.x for pcd in pcds], dim=0)
    f: torch.Tensor = torch.cat([pcd.f for pcd in pcds], dim=0)
    b: torch.Tensor = torch.cat([pcd.b for pcd in pcds], dim=0)
    return FeaturedPoints(x=x, f=f, b=b)

def _tuple_merge_featured_points(pcds: Tuple[FeaturedPoints]) -> FeaturedPoints:
    x: torch.Tensor = torch.cat([pcd.x for pcd in pcds], dim=0)
    f: torch.Tensor = torch.cat([pcd.f for pcd in pcds], dim=0)
    b: torch.Tensor = torch.cat([pcd.b for pcd in pcds], dim=0)
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


class GraphEdge(NamedTuple):
    edge_src: torch.Tensor # Position
    edge_dst: torch.Tensor # Feature
    edge_length: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    edge_scalars: Optional[torch.Tensor] = None
