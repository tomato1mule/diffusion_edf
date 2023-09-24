from typing import Tuple, List, Union, Optional, NamedTuple
from beartype import beartype

import torch
from e3nn import o3

from edf_interface.data import PointCloud
from edf_interface.data.pcd_utils import transform_points
from diffusion_edf.wigner import TransformFeatureQuaternion


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
        # assert w == ''
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
    def __init__(self, irreps: Optional[Union[str, o3.Irreps]]):
        super().__init__()
        if irreps is None:
            self.transform_features = None
        else:
            self.transform_features = TransformFeatureQuaternion(irreps=o3.Irreps(irreps))

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
    edge_weights: Optional[torch.Tensor] = None
    edge_logits: Optional[torch.Tensor] = None


@torch.jit.script
def set_graph_edge_attribute(graph_edge: GraphEdge, 
                             edge_src: Optional[torch.Tensor] = None, 
                             edge_dst: Optional[torch.Tensor] = None, 
                             edge_length: Union[str,Optional[torch.Tensor]] = '',
                             edge_attr: Union[str,Optional[torch.Tensor]] = '',
                             edge_scalars: Union[str,Optional[torch.Tensor]] = '',
                             edge_weights: Union[str,Optional[torch.Tensor]] = '',
                             edge_logits: Union[str,Optional[torch.Tensor]] = '',) -> GraphEdge:
    if edge_src is None:
        edge_src = graph_edge.edge_src
    if edge_dst is None:
        edge_dst = graph_edge.edge_dst
    if isinstance(edge_length, str):
        # assert edge_length == ''
        edge_length = graph_edge.edge_length
    if isinstance(edge_attr, str):
        # assert edge_attr == ''
        edge_attr = graph_edge.edge_attr
    if isinstance(edge_scalars, str):
        # assert edge_scalars == ''
        edge_scalars = graph_edge.edge_scalars
    if isinstance(edge_weights, str):
        # assert edge_weights == ''
        edge_weights = graph_edge.edge_weights
    if isinstance(edge_logits, str):
        # assert edge_logits == ''
        edge_logits = graph_edge.edge_logits
    

    return GraphEdge(edge_src=edge_src, 
                     edge_dst=edge_dst, 
                     edge_length=edge_length,
                     edge_attr=edge_attr,
                     edge_scalars=edge_scalars,
                     edge_weights=edge_weights,
                     edge_logits=edge_logits)

@torch.jit.script
def cat_graph_edges(graph_edge_1: GraphEdge, graph_edge_2: GraphEdge) -> GraphEdge:
    edge_src = torch.cat([graph_edge_1.edge_src, graph_edge_2.edge_src], dim=0)
    edge_dst = torch.cat([graph_edge_1.edge_dst, graph_edge_2.edge_dst], dim=0)

    edge_length_1, edge_length_2 = graph_edge_1.edge_length, graph_edge_2.edge_length
    if edge_length_1 is None or edge_length_2 is None:
        assert edge_length_1 is None and edge_length_2 is None
        edge_length = None
    else:
        assert isinstance(edge_length_1, torch.Tensor) and isinstance(edge_length_2, torch.Tensor)
        edge_length = torch.cat([edge_length_1, edge_length_2], dim=0)

    edge_attr_1, edge_attr_2 = graph_edge_1.edge_attr, graph_edge_2.edge_attr
    if edge_attr_1 is None or edge_attr_2 is None:
        assert edge_attr_1 is None and edge_attr_2 is None
        edge_attr = None
    else:
        assert isinstance(edge_attr_1, torch.Tensor) and isinstance(edge_attr_2, torch.Tensor)
        edge_attr = torch.cat([edge_attr_1, edge_attr_2], dim=0)

    edge_scalars_1, edge_scalars_2 = graph_edge_1.edge_scalars, graph_edge_2.edge_scalars
    if edge_scalars_1 is None or edge_scalars_2 is None:
        assert edge_scalars_1 is None and edge_scalars_2 is None
        edge_scalars = None
    else:
        assert isinstance(edge_scalars_1, torch.Tensor) and isinstance(edge_scalars_2, torch.Tensor)
        edge_scalars = torch.cat([edge_scalars_1, edge_scalars_2], dim=0)

    edge_weights_1, edge_weights_2 = graph_edge_1.edge_weights, graph_edge_2.edge_weights
    if edge_weights_1 is None or edge_weights_2 is None:
        assert edge_weights_1 is None and edge_weights_2 is None
        edge_weights = None
    else:
        assert isinstance(edge_weights_1, torch.Tensor) and isinstance(edge_weights_2, torch.Tensor)
        edge_weights = torch.cat([edge_weights_1, edge_weights_2], dim=0)

    edge_logits_1, edge_logits_2 = graph_edge_1.edge_logits, graph_edge_2.edge_logits
    if edge_logits_1 is None or edge_logits_2 is None:
        assert edge_logits_1 is None and edge_logits_2 is None
        edge_logits = None
    else:
        assert isinstance(edge_logits_1, torch.Tensor) and isinstance(edge_logits_2, torch.Tensor)
        edge_logits = torch.cat([edge_logits_1, edge_logits_2], dim=0)


    return GraphEdge(edge_src=edge_src, 
                     edge_dst=edge_dst, 
                     edge_length=edge_length,
                     edge_attr=edge_attr,
                     edge_scalars=edge_scalars,
                     edge_weights=edge_weights,
                     edge_logits=edge_logits)


@torch.jit.script
def cat_featured_points(fp1: FeaturedPoints, fp2: FeaturedPoints) -> FeaturedPoints:
    x = torch.cat([fp1.x, fp2.x], dim=0)
    f = torch.cat([fp1.f, fp2.f], dim=0)
    b = torch.cat([fp1.b, fp2.b], dim=0)

    w1, w2 = fp1.w, fp2.w
    if w1 is None or w2 is None:
        assert w1 is None and w2 is None
        w = None
    else:
        assert isinstance(w1, torch.Tensor) and isinstance(w2, torch.Tensor)
        w = torch.cat([w1, w2], dim=0)

    return FeaturedPoints(x=x, f=f, b=b, w=w)



