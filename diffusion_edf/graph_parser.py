import math
import warnings
from typing import Tuple, List, Dict, Optional, Union, Sequence
from beartype import beartype

import torch
from torch_cluster import radius_graph, radius, fps, graclus
from torch_scatter import scatter_add, scatter_mean

from e3nn import o3

from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge
from diffusion_edf.radial_func import soft_square_cutoff_2, SinusoidalPositionEmbeddings, BesselBasisEncoder
from diffusion_edf.irreps_utils import cutoff_irreps


class GraphEdgeEncoderBase(torch.nn.Module):
    r_cutoff: Tuple[Optional[float]]
    sh_dim: Optional[int]
    edge_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    nonscalar_sh_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    requires_length: bool
    requires_encoding: bool
    length_enc_dim: Optional[int]
    cutoff_eps: float

    @beartype
    def __init__(self, r_cutoff: Optional[Union[Union[float, int], Sequence[Union[float, int, None]]]],
                 irreps_sh: Optional[Union[str, o3.Irreps]],
                 length_enc_dim: Optional[int],
                 length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
                 length_enc_kwargs: Dict = {}, 
                 r_mincut_nonscalar_sh: Union[str, float, int, None] = 'default',  # Explicitly set this to ensure continuity of nonscalar spherical harmonics.
                 requires_length: Optional[bool] = None,  # Set to True if explicitly want length.
                 cutoff_eps: float = 1e-12):
        super().__init__()
        self.requires_length = True if requires_length else False
        self.cutoff_eps = cutoff_eps

        ######### Edge Cutoff Encoder #########
        # For continuity, information must vanish as edge length approaches maximum or minimum radius.
        if r_cutoff is None:
            self.edge_cutoff_ranges = None
        elif isinstance(r_cutoff, int) or isinstance(r_cutoff, float):
            self.edge_cutoff_ranges = (None, None, 0.8*float(r_cutoff), 1.0 * float(r_cutoff))
            self.requires_length = True
        elif isinstance(r_cutoff, Sequence):
            assert len(r_cutoff) == 4
            self.edge_cutoff_ranges = tuple(float(r) if (isinstance(r, float) or isinstance(r,int)) else r for r in r_cutoff)
            self.requires_length = True
        else:
            raise TypeError(f"Unknown type of r_cutoff: {r_cutoff}")
        

        ######### nonscalar spherical harmonics cutoff #########
        if self.edge_cutoff_ranges is None:
            if r_mincut_nonscalar_sh == 'default':
                raise ValueError(f"Please explicitly set r_mincut_nonscalar_sh to either a number or None")
        elif self.edge_cutoff_ranges[0] is None or self.edge_cutoff_ranges[1] is None:
            assert self.edge_cutoff_ranges[0] is None and self.edge_cutoff_ranges[1] is None, f"Wrong ranges armument: {self.edge_cutoff_ranges}"
            if r_mincut_nonscalar_sh == 'default':
                r_mincut_nonscalar_sh = None
            elif r_mincut_nonscalar_sh is None:
                pass
            elif isinstance(r_mincut_nonscalar_sh, int) or isinstance(r_mincut_nonscalar_sh, float):
                r_mincut_nonscalar_sh = float(r_mincut_nonscalar_sh)
                warnings.warn(f"r_mincut_nonscalar_sh ({r_mincut_nonscalar_sh}) and self.edge_cutoff_ranges[0:1] ({self.edge_cutoff_ranges}) are simultaneously set. Are you sure?")
            else:
                raise TypeError(f"Unknown type of r_mincut_nonscalar_sh: {r_mincut_nonscalar_sh}")
            
        if isinstance(r_mincut_nonscalar_sh, int) or isinstance(r_mincut_nonscalar_sh, float):
            self.nonscalar_sh_cutoff_ranges = (0.2*float(r_mincut_nonscalar_sh), 1.0*float(r_mincut_nonscalar_sh), None, None)
            self.requires_length = True
        elif r_mincut_nonscalar_sh is None:
            self.nonscalar_sh_cutoff_ranges = None
        else:
            raise TypeError(f"Unknown type of nonscalar_sh_cutoff_ranges: {self.nonscalar_sh_cutoff_ranges}")

        ######### Spherical Harmonics Encoder #########
        if irreps_sh is None:
            self.irreps_sh = None
            self.sh_dim = None
            self.sh = None
        else:
            self.irreps_sh = o3.Irreps(irreps_sh)
            self.sh_dim = self.irreps_sh.dim
            self.sh = o3.SphericalHarmonics(irreps_out = self.irreps_sh, normalize = True, normalization='component')

        ######### Length Encoder #########
        if length_enc_dim is None:
            if length_enc_type is not None:
                raise ValueError(f"length_enc_type: {length_enc_type} must be explicitly set to None if length_enc_dim is None.")
            self.length_enc = None
            self.length_enc_dim = None
        else:
            assert length_enc_type is not None
            self.requires_length = True
            self.length_enc_dim = length_enc_dim
            if length_enc_type == 'SinusoidalPositionEmbeddings':
                if self.edge_cutoff_ranges is not None:
                    if self.edge_cutoff_ranges[-1] is not None:
                        if 'max_val' not in length_enc_kwargs.keys():
                            length_enc_kwargs['max_val'] = self.edge_cutoff_ranges[-1]
                self.length_enc = SinusoidalPositionEmbeddings(dim=self.length_enc_dim, **length_enc_kwargs)
            elif length_enc_type == 'BesselBasisEncoder':
                if self.edge_cutoff_ranges is not None:
                    if self.edge_cutoff_ranges[-1] is not None:
                        if 'max_val' not in length_enc_kwargs.keys():
                            length_enc_kwargs['max_val'] = self.edge_cutoff_ranges[-1]
                        if 'max_cutoff' not in length_enc_kwargs.keys():
                            length_enc_kwargs['max_cutoff'] = True
                    if self.edge_cutoff_ranges[0] is not None:
                        if 'min_val' not in length_enc_kwargs.keys():
                            length_enc_kwargs['min_val'] = self.edge_cutoff_ranges[0]
                self.length_enc = BesselBasisEncoder(dim=self.length_enc_dim, **length_enc_dim)
            else:
                raise ValueError(f"Unknown length encoder type: {length_enc_kwargs['type']}")
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
        if self.edge_cutoff_ranges is None:
            edge_cutoff = None
        else:
            edge_cutoff = soft_square_cutoff_2(x=edge_length, ranges=self.edge_cutoff_ranges) # (Nedge, )
        if self.nonscalar_sh_cutoff_ranges is None:
            cutoff_nonscalar = None
        else:
            cutoff_nonscalar = soft_square_cutoff_2(x=edge_length, ranges=self.nonscalar_sh_cutoff_ranges) # (Nedge, )

        if self.length_enc is not None:
            edge_scalars = self.length_enc(edge_length) # (Nedge, D)
        else:
            edge_scalars = None
        
        if self.sh is not None:
            edge_sh = self.sh(edge_vec)                 # (Nedge, Y)
        else:
            edge_sh = None

        if isinstance(edge_sh, torch.Tensor):
            edge_sh = cutoff_irreps(f=edge_sh, 
                                    edge_cutoff=None,
                                    cutoff_scalar=None, 
                                    cutoff_nonscalar=cutoff_nonscalar,
                                    irreps=self.irreps_sh)
                
        if edge_cutoff is None:
            log_edge_cutoff = None
        else:
            log_edge_cutoff = torch.log(edge_cutoff + self.cutoff_eps)

        return GraphEdge(edge_src=edge_src, 
                         edge_dst=edge_dst, 
                         edge_attr=edge_sh, 
                         edge_length=edge_length, 
                         edge_scalars=edge_scalars, 
                         edge_weights=edge_cutoff,
                         edge_logits=log_edge_cutoff,)


class InfiniteBipartite(GraphEdgeEncoderBase):
    r_cutoff: Tuple[Optional[float]]
    sh_dim: Optional[int]
    edge_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    nonscalar_sh_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    requires_length: bool
    requires_encoding: bool
    length_enc_dim: Optional[int]
    cutoff_eps: float

    




class RadiusBipartite(GraphEdgeEncoderBase):
    max_neighbors: Optional[int]
    r_cluster: Optional[float]
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
    cutoff_sh: bool
    cutoff_eps: float

    @beartype
    def __init__(self, r_cluster: Optional[float],
                 r_mincut_nonscalar: Optional[float],  # Need to set non-zero r_mincut for continuity
                 length_enc_dim: Optional[int],
                 r_maxcut: Union[Optional[float], str] = 'default',
                 length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
                 length_enc_kwarg: Dict = {}, 
                 sh_irreps: Optional[Union[str, o3.Irreps]] = None,
                 r_mincut_scalar: Optional[float] = None,
                 requires_length: Optional[bool] = None,
                 max_neighbors: Optional[int] = 1000,
                 cutoff_sh: bool = False,
                 cutoff_eps: float = 1e-12):
        
        self.max_neighbors = max_neighbors
        self.r_cluster = r_cluster
        if r_maxcut == 'default':
            if r_cluster is None:
                r_maxcut = None
            else:
                raise ValueError("r_maxcut must be explicitly set to Optional[float] if r_cluster is not None")

        super().__init__(r_maxcut=r_maxcut,
                         r_mincut_nonscalar=r_mincut_nonscalar,
                         length_enc_dim=length_enc_dim,
                         length_enc_type=length_enc_type,
                         length_enc_kwarg=length_enc_kwarg,
                         sh_irreps=sh_irreps,
                         r_mincut_scalar=r_mincut_scalar,
                         requires_length=requires_length,
                         cutoff_sh=cutoff_sh,
                         cutoff_eps=cutoff_eps)

    def forward(self, src: FeaturedPoints, dst: FeaturedPoints, max_neighbors: Optional[int] = None) -> GraphEdge:
        assert src.x.ndim == 2
        assert dst.x.ndim == 2
        if max_neighbors is None:
            if self.max_neighbors is None:
                raise ValueError("max_neighbor must be specified")
            else:
                max_neighbors = self.max_neighbors
        assert max_neighbors is not None
        if self.r_cluster is None:
            edge_src, edge_dst = torch.meshgrid(torch.arange(len(src.x), device = src.x.device), torch.arange(len(dst.x), device = dst.x.device), indexing='ij')
            edge_src = edge_src.reshape(-1)
            edge_dst = edge_dst.reshape(-1)
        else:
            r = self.r_cluster
            assert isinstance(r, float)
            edge = radius(x = src.x, y = dst.x, r=r, batch_x=src.b, batch_y=dst.b, max_num_neighbors=max_neighbors)
            edge_dst, edge_src = edge[0], edge[1]

        if not self.requires_encoding:
            return GraphEdge(edge_src=edge_src, edge_dst=edge_dst)
        
        return self._encode_edges(x_src=src.x, x_dst=dst.x, edge_src=edge_src, edge_dst=edge_dst)