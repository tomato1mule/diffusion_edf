import math
from typing import Tuple, List, Dict, Optional, Union
from beartype import beartype

import torch
from torch_cluster import radius_graph, radius, fps, graclus
from torch_scatter import scatter_add, scatter_mean

from e3nn import o3

from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge
from diffusion_edf.radial_func import SinusoidalPositionEmbeddings, soft_square_cutoff_2


class GraphEdgeEncoderBase(torch.nn.Module):
    """
    length_enc_kwarg: length_enc_kwarg: {'n': 10000}
    """
    r: float
    max_neighbors: int
    sh_irreps: Optional[o3.Irreps]
    sh_dim: int
    cutoff: bool
    scalar_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    nonscalar_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    encode_graph: bool

    @beartype
    def __init__(self, r: float, 
                 length_enc_dim: Optional[int],
                 length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
                 length_enc_kwarg: Dict = {}, 
                 sh_irreps: Optional[Union[str, o3.Irreps]] = None,
                 cutoff: bool = True):
        super().__init__()
        self.r = r

        ######### Cutoff Encoder #########
        self.cutoff = cutoff
        if self.cutoff is False:
            self.scalar_cutoff_ranges = None
            self.nonscalar_cutoff_ranges = None
        else:
            self.scalar_cutoff_ranges = (None, None, 0.8 * self.r, 0.99 * self.r)                      # For continuity, information must vanish as edge length approaches maximum raidus.
            self.nonscalar_cutoff_ranges = (0.05 * self.r, 0.15 * self.r, 0.8 * self.r, 0.99 * self.r) # Spherical Harmonics of degree >1 are sigular at zero, so must be cut-off for continuity

        ######### Spherical Harmonics Encoder #########
        if sh_irreps is not None:
            self.sh_irreps = o3.Irreps(sh_irreps)
            self.sh_dim = self.sh_irreps.dim
            self.sh = o3.SphericalHarmonics(irreps_out = self.sh_irreps, normalize = True, normalization='component')
        else:
            self.sh_irreps = None
            self.sh = None
            self.sh_dim = 0
        
        ######### Length Encoder #########
        if length_enc_type is None or length_enc_dim is None:
            assert length_enc_type is None and length_enc_dim is None
            self.length_enc = None
            self.length_enc_dim = None
        else:
            self.length_enc_dim = length_enc_dim
            if length_enc_type == 'SinusoidalPositionEmbeddings':
                if 'max_val' not in length_enc_kwarg.keys():
                    length_enc_kwarg['max_val'] = self.r
                self.length_enc = SinusoidalPositionEmbeddings(dim=self.length_enc_dim, **length_enc_kwarg)
            else:
                raise ValueError(f"Unknown length encoder type: {length_enc_kwarg['type']}")
            
        ##################################
        if self.sh is not None or \
           self.scalar_cutoff_ranges is not None or \
           self.nonscalar_cutoff_ranges is None or \
           self.length_enc is not None:
           self.encode_graph = True
        else:
           self.encode_graph = False

            
    def _encode_edges(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor) -> GraphEdge:
        if not self.encode_graph:
            raise ValueError("You don't have to encode the graph.")
        
        assert x_src.ndim == 2
        assert x_dst.ndim == 2
        assert edge_src.ndim == 1
        assert edge_dst.ndim == 1

        edge_vec = x_src.index_select(0, edge_src) - x_dst.index_select(0, edge_dst) # (Nedge, 3)
        edge_length = edge_vec.norm(dim=1, p=2)                                      # (Nedge, )
        if self.scalar_cutoff_ranges is None:
            cutoff_scalar = None
        else:
            cutoff_scalar = soft_square_cutoff_2(x=edge_length, ranges=self.scalar_cutoff_ranges) # (Nedge, )
        if self.nonscalar_cutoff_ranges is None:
            cutoff_nonscalar = None
        else:
            cutoff_nonscalar = soft_square_cutoff_2(x=edge_length, ranges=self.nonscalar_cutoff_ranges) # (Nedge, )

        if self.length_enc is not None:
            edge_scalars = self.length_enc(edge_length) # (Nedge, D)
        else:
            edge_scalars = None
        
        if self.sh is not None:
            edge_sh = self.sh(edge_vec)                 # (Nedge, Y)
        else:
            edge_sh = None

        if edge_sh is not None and self.cutoff is True:
            edge_sh_cutoff = []
            last_idx = 0
            for n, (l,p) in self.sh_irreps:
                d = n * (2*l + 1)
                if l == 0:
                    edge_sh_cutoff.append(
                        edge_sh[..., last_idx: last_idx+d] * cutoff_scalar[..., None]
                    )
                else:
                    edge_sh_cutoff.append(
                        edge_sh[..., last_idx: last_idx+d] * cutoff_nonscalar[..., None]
                    )
                
                last_idx = last_idx + d

            edge_sh_cutoff = torch.cat(edge_sh_cutoff, dim=-1)

        else:
            edge_sh_cutoff = edge_sh
                
        return GraphEdge(edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh_cutoff, edge_length=edge_length, edge_scalars=edge_scalars, edge_weight_scalar=cutoff_scalar, edge_weight_nonscalar=cutoff_nonscalar)


class RadiusBipartite(GraphEdgeEncoderBase):
    """
    length_enc_kwarg: {'n': 10000}
    """
    max_neighbors: Optional[int]

    @beartype
    def __init__(self, r: float, 
                 length_enc_dim: Optional[int],
                 length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
                 length_enc_kwarg: Dict = {}, 
                 sh_irreps: Optional[Union[str, o3.Irreps]] = None,
                 cutoff: bool = True,
                 max_neighbors: int = 1000):
        super().__init__(r=r, length_enc_dim=length_enc_dim, length_enc_type=length_enc_type, length_enc_kwarg=length_enc_kwarg, sh_irreps=sh_irreps, cutoff=cutoff)
        self.max_neighbors = max_neighbors

    def forward(self, src: FeaturedPoints, dst: FeaturedPoints, max_neighbors: Optional[int] = None) -> GraphEdge:
        if max_neighbors is None:
            if self.max_neighbors is None:
                raise ValueError("max_neighbor must be specified")
            else:
                max_neighbors = self.max_neighbors
        assert max_neighbors is not None
        edge = radius(x = src.x, y = dst.x, r=self.r, batch_x=src.b, batch_y=dst.b, max_num_neighbors=max_neighbors)
        edge_dst, edge_src = edge[0], edge[1]
        if not self.encode_graph:
            return GraphEdge(edge_src=edge_src, edge_dst=edge_dst)
        
        return self._encode_edges(x_src=src.x, x_dst=dst.x, edge_src=edge_src, edge_dst=edge_dst)