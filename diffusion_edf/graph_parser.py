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
    r_maxcut: Optional[float]
    r_mincut_nonscalar: Optional[float]
    r_mincut_scalar: Optional[float]
    #sh_irreps: Optional[o3.Irreps]
    sh_dim: Optional[int]
    scalar_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    nonscalar_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    requires_length: bool
    requires_encoding: bool
    length_enc_dim: Optional[int]

    @beartype
    def __init__(self, r_maxcut: Optional[float],
                 r_mincut_nonscalar: Optional[float], 
                 length_enc_dim: Optional[int],
                 length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
                 length_enc_kwarg: Dict = {}, 
                 sh_irreps: Optional[Union[str, o3.Irreps]] = None,
                 r_mincut_scalar: Optional[float] = None,
                 requires_length: Optional[bool] = None):
        super().__init__()
        self.r_maxcut = r_maxcut
        self.r_mincut_nonscalar = r_mincut_nonscalar
        self.r_mincut_scalar = r_mincut_scalar
        self.requires_length = True if requires_length else False

        ######### Cutoff Encoder #########
        # For continuity, information must vanish as edge length approaches maximum raidus.
        # Spherical Harmonics of degree >1 are sigular at zero, so must be cut-off for continuity
        if self.r_maxcut is None:
            if self.r_mincut_scalar is None:
                self.scalar_cutoff_ranges = None
            else:
                self.scalar_cutoff_ranges = (0.2*self.r_mincut_scalar, 1.0*self.r_mincut_scalar, None, None)
                self.requires_length = True

            if self.r_mincut_nonscalar is None:
                self.nonscalar_cutoff_ranges = None
            else:
                self.nonscalar_cutoff_ranges = (0.2*self.r_mincut_nonscalar, 1.0*self.r_mincut_nonscalar, None, None)
                self.requires_length = True
        else:
            if self.r_mincut_scalar is None:
                self.scalar_cutoff_ranges = (None, None, 0.8*self.r_maxcut, 0.99*self.r_maxcut)
                self.requires_length = True
            else:
                self.scalar_cutoff_ranges = (0.2*self.r_mincut_scalar, 1.0*self.r_mincut_scalar, 0.8*self.r_maxcut, 0.99*self.r_maxcut)
                self.requires_length = True
                
            if self.r_mincut_nonscalar is None:
                self.nonscalar_cutoff_ranges = (None, None, 0.8*self.r_maxcut, 0.99*self.r_maxcut)
                self.requires_length = True
            else:
                self.nonscalar_cutoff_ranges = (0.2*self.r_mincut_nonscalar, 1.0*self.r_mincut_nonscalar, 0.8*self.r_maxcut, 0.99*self.r_maxcut)
                self.requires_length = True

        ######### Spherical Harmonics Encoder #########
        if sh_irreps is None:
            self.sh_irreps = None
            self.sh_dim = None
            self.sh = None
        else:
            self.sh_irreps = o3.Irreps(sh_irreps)
            self.sh_dim = self.sh_irreps.dim
            self.sh = o3.SphericalHarmonics(irreps_out = self.sh_irreps, normalize = True, normalization='component')

        ######### Length Encoder #########
        if length_enc_dim is None:
            assert length_enc_type is None
            self.length_enc = None
            self.length_enc_dim = None
        else:
            assert length_enc_type is not None
            self.requires_length = True
            self.length_enc_dim = length_enc_dim
            if length_enc_type == 'SinusoidalPositionEmbeddings':
                if 'max_val' not in length_enc_kwarg.keys():
                    length_enc_kwarg['max_val'] = self.r_maxcut
                self.length_enc = SinusoidalPositionEmbeddings(dim=self.length_enc_dim, **length_enc_kwarg)
            else:
                raise ValueError(f"Unknown length encoder type: {length_enc_kwarg['type']}")
        ##################################
        if requires_length is False and requires_length != self.requires_length:
            raise ValueError(f"requires_length is manually set to False, but it seems length is required")
        if self.sh is None and not self.requires_length:
            self.requires_encoding = False
        else:
            self.requires_encoding = True
            
    def _encode_edges(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor) -> GraphEdge:
        if not self.requires_encoding:
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

        if edge_sh is not None:
            edge_sh_cutoff = []
            last_idx = 0
            for n, (l,p) in self.sh_irreps:
                d = n * (2*l + 1)
                if l == 0 and cutoff_scalar is not None:
                    edge_sh_cutoff.append(
                        edge_sh[..., last_idx: last_idx+d] * cutoff_scalar[..., None]
                    )
                elif l != 0 and cutoff_nonscalar is not None:
                    edge_sh_cutoff.append(
                        edge_sh[..., last_idx: last_idx+d] * cutoff_nonscalar[..., None]
                    )
                else:
                    edge_sh_cutoff.append(edge_sh[..., last_idx: last_idx+d])
                
                last_idx = last_idx + d

            edge_sh_cutoff = torch.cat(edge_sh_cutoff, dim=-1)

        else:
            edge_sh_cutoff = edge_sh
                
        return GraphEdge(edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh_cutoff, edge_length=edge_length, edge_scalars=edge_scalars, edge_weight_scalar=cutoff_scalar, edge_weight_nonscalar=cutoff_nonscalar)


class RadiusBipartite(GraphEdgeEncoderBase):
    max_neighbors: Optional[int]
    r_maxcut: float
    r_mincut_nonscalar: Optional[float]
    r_mincut_scalar: Optional[float]
    #sh_irreps: Optional[o3.Irreps]
    sh_dim: Optional[int]
    scalar_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    nonscalar_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    requires_length: bool
    requires_encoding: bool
    length_enc_dim: Optional[int]

    @beartype
    def __init__(self, r_maxcut: float,
                 r_mincut_nonscalar: Optional[float], 
                 length_enc_dim: Optional[int],
                 length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
                 length_enc_kwarg: Dict = {}, 
                 sh_irreps: Optional[Union[str, o3.Irreps]] = None,
                 r_mincut_scalar: Optional[float] = None,
                 requires_length: Optional[bool] = None,
                 max_neighbors: Optional[int] = 1000):
        super().__init__(r_maxcut=r_maxcut,
                         r_mincut_nonscalar=r_mincut_nonscalar,
                         length_enc_dim=length_enc_dim,
                         length_enc_type=length_enc_type,
                         length_enc_kwarg=length_enc_kwarg,
                         sh_irreps=sh_irreps,
                         r_mincut_scalar=r_mincut_scalar,
                         requires_length=requires_length)
        self.max_neighbors = max_neighbors

    def forward(self, src: FeaturedPoints, dst: FeaturedPoints, max_neighbors: Optional[int] = None) -> GraphEdge:
        if max_neighbors is None:
            if self.max_neighbors is None:
                raise ValueError("max_neighbor must be specified")
            else:
                max_neighbors = self.max_neighbors
        assert max_neighbors is not None
        edge = radius(x = src.x, y = dst.x, r=self.r_maxcut, batch_x=src.b, batch_y=dst.b, max_num_neighbors=max_neighbors)
        edge_dst, edge_src = edge[0], edge[1]
        if not self.requires_encoding:
            return GraphEdge(edge_src=edge_src, edge_dst=edge_dst)
        
        return self._encode_edges(x_src=src.x, x_dst=dst.x, edge_src=edge_src, edge_dst=edge_dst)