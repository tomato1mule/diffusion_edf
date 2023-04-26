from typing import Union, Optional, List, Tuple, Dict
import math

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_cluster import radius
from torch_scatter import scatter_sum

from diffusion_edf.so3_util import ParityInversionSh, multiply_irreps, sort_irreps_even_first
    

# Borrowed from https://github.com/lucidrains/denoising-diffusion-pytorch
class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim: int, max_t: float = 1., n: float = 10000.):
        super().__init__()
        self.dim = dim
        self.n = n
        self.max_t = max_t

    def forward(self, time: torch.Tensor):
        assert time.ndim == 1, f"{time.shape}"
        time = time / self.max_t * self.n # time: 0~10000

        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.n) / (half_dim - 1) # Period: 2pi~10000*2pi
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # shape: (self.dim/2, )
        embeddings = time[:, None] * embeddings[None, :]                            # shape: (nBatch, self.dim/2) 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)        # shape: (nBatch, self.dim)

        return embeddings
    


def sample_reference_points(src_points: torch.Tensor, dst_points: torch.Tensor, r: float, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_dst, edge_src = radius(x=src_points, y=dst_points, r=r)
    n_points = len(dst_points)
    n_neighbor = scatter_sum(src=torch.ones_like(edge_dst), index=edge_dst, dim_size=n_points)
    total_count = n_neighbor.sum()
    if total_count <= 0:
        raise ValueError("There is no connected edges. Increase the clustering radius.")
    p_choice = n_neighbor / total_count

    sampled_idx = torch.multinomial(p_choice, num_samples=n_samples)
    return dst_points.index_select(0, sampled_idx), n_neighbor