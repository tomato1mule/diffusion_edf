from typing import Tuple, List, Dict, Optional, Union

import torch
from torch_cluster import radius_graph, radius, fps, graclus
from torch_scatter import scatter_add, scatter_mean


class RadiusGraph(torch.nn.Module):
    def __init__(self, r: float, max_num_neighbors: int):
        super().__init__()
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors

    def forward(self, node_coord_src: torch.Tensor, node_feature_src: torch.Tensor, batch_src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3

        node_coord_dst = node_coord_src
        batch_dst = batch_src
        node_feature_dst = node_feature_src
        N_nodes = len(node_coord_dst)

        edge = radius_graph(node_coord_dst, r=self.r, batch=batch_dst, loop=False, max_num_neighbors=self.max_num_neighbors)
        edge_dst = edge[0]
        edge_src = edge[1]
        degree = scatter_add(src = torch.ones_like(edge_dst), index = edge_dst, dim=0, dim_size=N_nodes)

        
        return node_feature_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst
    


class RadiusConnect(torch.nn.Module):
    def __init__(self, r: float, max_num_neighbors: int, offset: Optional[float] = None):
        super().__init__()
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors
        if offset is not None:
            raise NotImplementedError
        self.offset = offset

    def forward(self, node_coord_src: torch.Tensor, batch_src: torch.Tensor, node_coord_dst: torch.Tensor, batch_dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge = radius(x = node_coord_src, y = node_coord_dst, r=self.r, batch_x=batch_src, batch_y=batch_dst, max_num_neighbors=self.max_num_neighbors)
        edge_dst = edge[0]
        edge_src = edge[1]

        return edge_src, edge_dst
    


class FpsPool(torch.nn.Module):
    def __init__(self, ratio: float, random_start: bool, r: float, max_num_neighbors: int):
        super().__init__()
        self.ratio: float = ratio
        self.random_start: bool = random_start
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors
        self.radius_connect = RadiusConnect(r=self.r, max_num_neighbors=self.max_num_neighbors)

    def forward(self, node_coord_src: torch.Tensor, node_feature_src: torch.Tensor, batch_src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3
        
        node_dst_idx = fps(src=node_coord_src, batch=batch_src, ratio=self.ratio, random_start=self.random_start)
        node_coord_dst = node_coord_src.index_select(index=node_dst_idx, dim=0)
        batch_dst = batch_src.index_select(index=node_dst_idx, dim=0)
        N_nodes = len(node_dst_idx)

        edge_src, edge_dst = self.radius_connect(node_coord_src=node_coord_src, node_coord_dst=node_coord_dst, batch_src=batch_src, batch_dst=batch_dst)

        non_self_idx = (node_dst_idx[edge_dst] != edge_src).nonzero().squeeze(-1)
        edge_src = edge_src[non_self_idx]
        edge_dst = edge_dst[non_self_idx]

        degree = scatter_add(src = torch.ones_like(edge_dst), index = edge_dst, dim=0, dim_size=N_nodes)
        #node_feature_dst = scatter_mean(src = node_feature_src.index_select(index=edge_src, dim=0), index = edge_dst, dim=0, dim_size=N_nodes)
        node_feature_dst = node_feature_src.index_select(index=node_dst_idx, dim=0)

        return node_feature_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst