import math
import warnings
from typing import Tuple, List, Dict, Optional, Union, Sequence
from beartype import beartype

import torch
from torch_cluster import radius_graph, radius, fps, graclus
from torch_scatter import scatter_add, scatter_mean

from e3nn import o3

from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge
from diffusion_edf.radial_func import soft_square_cutoff_2, SinusoidalPositionEmbeddings, BesselBasisEncoder, GaussianRadialBasis
from diffusion_edf.irreps_utils import cutoff_irreps


class GraphEdgeEncoderBase(torch.nn.Module):
    r_cutoff: Tuple[Optional[float]]
    sh_dim: Optional[int]
    edge_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    nonscalar_sh_cutoff_ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]
    requires_length: bool
    requires_encoding: bool
    offset: Optional[float]
    sh_cutoff: bool

    @beartype
    def __init__(self, r_cutoff: Optional[Union[Union[float, int], Sequence[Union[float, int, None]]]],
                 irreps_sh: Optional[Union[str, o3.Irreps]],
                 length_enc: Optional[torch.nn.Module],
                 r_mincut_nonscalar_sh: Union[str, float, int, None] = 'default',  # Explicitly set this to ensure continuity of nonscalar spherical harmonics.
                 requires_length: Optional[bool] = None,  # Set to True if explicitly want length.
                 cutoff_eps: float = 1e-12,
                 sh_cutoff: bool = False):
        super().__init__()
        self.requires_length = True if requires_length else False
        self.register_buffer('cutoff_eps', torch.tensor(cutoff_eps))

        ######### Length Encoder ##############
        self.length_enc = length_enc
        if self.length_enc is not None:
            self.requires_length = True
        # if length_enc_dim is None:
        #     if length_enc_type is not None:
        #         raise ValueError(f"length_enc_type: {length_enc_type} must be explicitly set to None if length_enc_dim is None.")
        #     self.length_enc = None
        #     self.length_enc_dim = None
        # else:
        #     assert length_enc_type is not None
        #     self.requires_length = True
        #     self.length_enc_dim = length_enc_dim
        #     if length_enc_type == 'SinusoidalPositionEmbeddings':
        #         if self.edge_cutoff_ranges is not None:
        #             if self.edge_cutoff_ranges[-1] is not None:
        #                 if 'max_val' not in length_enc_kwargs.keys():
        #                     length_enc_kwargs['max_val'] = self.edge_cutoff_ranges[-1]
        #         self.length_enc = SinusoidalPositionEmbeddings(dim=self.length_enc_dim, **length_enc_kwargs)
        #     elif length_enc_type == 'BesselBasisEncoder':
        #         if self.edge_cutoff_ranges is not None:
        #             if self.edge_cutoff_ranges[-1] is not None:
        #                 if 'max_val' not in length_enc_kwargs.keys():
        #                     length_enc_kwargs['max_val'] = self.edge_cutoff_ranges[-1]
        #                 if 'max_cutoff' not in length_enc_kwargs.keys():
        #                     length_enc_kwargs['max_cutoff'] = True
        #             if self.edge_cutoff_ranges[0] is not None:
        #                 if 'min_val' not in length_enc_kwargs.keys():
        #                     length_enc_kwargs['min_val'] = self.edge_cutoff_ranges[0]
        #         self.length_enc = BesselBasisEncoder(dim=self.length_enc_dim, **length_enc_dim)
        #     else:
        #         raise ValueError(f"Unknown length encoder type: {length_enc_kwargs['type']}")

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
        
        self.offset = None
        if self.edge_cutoff_ranges is not None:
            if self.edge_cutoff_ranges[0] is not None:
                self.offset = float(self.edge_cutoff_ranges[0])
        self.sh_cutoff = sh_cutoff

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
            else:
                raise TypeError(f"Unknown type of r_mincut_nonscalar_sh: {r_mincut_nonscalar_sh}")
        else:
            if r_mincut_nonscalar_sh == 'default':
                r_mincut_nonscalar_sh = None
            elif r_mincut_nonscalar_sh is None:
                pass
            elif isinstance(r_mincut_nonscalar_sh, int) or isinstance(r_mincut_nonscalar_sh, float):
                r_mincut_nonscalar_sh = float(r_mincut_nonscalar_sh)
                # warnings.warn(f"r_mincut_nonscalar_sh ({r_mincut_nonscalar_sh}) and self.edge_cutoff_ranges[0:1] ({self.edge_cutoff_ranges}) are simultaneously set. Are you sure?")
                raise ValueError(f"r_mincut_nonscalar_sh ({r_mincut_nonscalar_sh}) and self.edge_cutoff_ranges[0:1] ({self.edge_cutoff_ranges}) are simultaneously set. Are you sure?")
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
        
        ##################################
        if requires_length is False and requires_length != self.requires_length:
            raise ValueError(f"requires_length is manually set to False, but it seems length is required")
        if self.sh is None and not self.requires_length:
            self.requires_encoding = False
        else:
            self.requires_encoding = True
            
    # @torch.autocast(device_type='cuda', enabled=False)
    def _encode_edges(self, x_src: torch.Tensor, 
                      x_dst: torch.Tensor, 
                      edge_src: torch.Tensor, 
                      edge_dst: torch.Tensor,
                      fill_edge_weights: Optional[float] = None) -> GraphEdge:
        if not self.requires_encoding:
            raise ValueError("You don't have to encode the graph.")
        
        assert x_src.ndim == 2
        assert x_dst.ndim == 2
        assert edge_src.ndim == 1
        assert edge_dst.ndim == 1

        edge_vec = x_src.index_select(0, edge_src) - x_dst.index_select(0, edge_dst) # (Nedge, 3)
        edge_length = edge_vec.norm(dim=1, p=2)                                      # (Nedge, )

        offset = self.offset
        if offset is not None:
            in_range_idx = (edge_length >= offset).nonzero().squeeze(-1)
            edge_src, edge_dst, edge_vec, edge_length = edge_src[in_range_idx], edge_dst[in_range_idx], edge_vec[in_range_idx], edge_length[in_range_idx]

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
            if self.sh_cutoff:
                edge_sh = cutoff_irreps(f=edge_sh, 
                                        edge_cutoff=edge_cutoff,
                                        cutoff_scalar=None, 
                                        cutoff_nonscalar=cutoff_nonscalar,
                                        irreps=self.irreps_sh)
            else:
                edge_sh = cutoff_irreps(f=edge_sh, 
                                        edge_cutoff=None,
                                        cutoff_scalar=None, 
                                        cutoff_nonscalar=cutoff_nonscalar,
                                        irreps=self.irreps_sh)
                
        if edge_cutoff is None:
            if fill_edge_weights is None:
                edge_cutoff_ = None
                log_edge_cutoff = None
            else:
                assert isinstance(fill_edge_weights, float), f"{fill_edge_weights}"
                # edge_cutoff_ = torch.empty_like(edge_length).fill_(fill_edge_weights)                # torch.jit.script incompatible
                # log_edge_cutoff = torch.empty_like(edge_length).fill_(math.log(fill_edge_weights))   # torch.jit.script incompatible
                edge_cutoff_ = torch.ones_like(edge_length) * fill_edge_weights
                log_edge_cutoff = torch.ones_like(edge_length) * math.log(fill_edge_weights)
        else:
            if edge_cutoff.requires_grad:
                edge_cutoff = torch.where(edge_cutoff >= self.cutoff_eps, edge_cutoff, self.cutoff_eps + (edge_cutoff - edge_cutoff.detach())) # Straight-through gradient estimation trick
            else:
                edge_cutoff = torch.max(edge_cutoff, self.cutoff_eps)
            log_edge_cutoff = torch.log(edge_cutoff)
            edge_cutoff_ = edge_cutoff # This stupid code is due to torch.jit.script compatibility.

        return GraphEdge(edge_src=edge_src, 
                         edge_dst=edge_dst, 
                         edge_attr=edge_sh, 
                         edge_length=edge_length, 
                         edge_scalars=edge_scalars, 
                         edge_weights=edge_cutoff_,
                         edge_logits=log_edge_cutoff,)




class InfiniteBipartite(GraphEdgeEncoderBase):
    fill_edge_weights: Optional[float]
    @beartype
    def __init__(self, irreps_sh: Optional[Union[str, o3.Irreps]],
                 r_mincut_nonscalar_sh: Optional[Union[float, int]],
                 length_enc_dim: Optional[int],
                 length_enc_max_r: Optional[Union[float, int]] = None,
                 length_enc_type: Optional[str] = 'SinusoidalPositionEmbeddings',
                 sh_cutoff: bool = False,
                 fill_edge_weights: bool = False,
                 ):
        if length_enc_dim is None:
            length_enc = None
        else:
            assert length_enc_max_r is not None
            if length_enc_type == 'SinusoidalPositionEmbeddings':
                length_enc = SinusoidalPositionEmbeddings(dim = length_enc_dim, 
                                                          max_val=float(length_enc_max_r),
                                                          n=1000.)
            elif length_enc_type == 'BesselBasisEncoder':
                length_enc = torch.nn.Sequential(
                    BesselBasisEncoder(dim = 8,
                                       max_val = float(length_enc_max_r),
                                       min_val = 0.,
                                       max_cutoff = False),
                    torch.nn.Linear(8, length_enc_dim)
                )
            elif length_enc_type == 'GaussianRadialBasis':
                length_enc = GaussianRadialBasis(dim = length_enc_dim, max_val=float(length_enc_max_r))
            else:
                raise ValueError(f"Unknown length encoder type: {length_enc_type}")
        
        if fill_edge_weights:
            self.fill_edge_weights = 1.
        else:
            self.fill_edge_weights = None
        
        super().__init__(r_cutoff=None, 
                         irreps_sh=irreps_sh,
                         r_mincut_nonscalar_sh=r_mincut_nonscalar_sh,
                         length_enc=length_enc,
                         sh_cutoff=sh_cutoff)
        
    def forward(self, src: FeaturedPoints, 
                dst: FeaturedPoints, 
                max_neighbors: Optional[int] = None       # just a placeholder
                ) -> GraphEdge:
        assert src.x.ndim == 2
        assert dst.x.ndim == 2

        edge_src, edge_dst = torch.meshgrid(torch.arange(len(src.x), device = src.x.device), torch.arange(len(dst.x), device = dst.x.device), indexing='ij')
        edge_src = edge_src.reshape(-1)
        edge_dst = edge_dst.reshape(-1)

        if not self.requires_encoding:
            return GraphEdge(edge_src=edge_src, edge_dst=edge_dst)
        
        return self._encode_edges(x_src=src.x, x_dst=dst.x, edge_src=edge_src, edge_dst=edge_dst, fill_edge_weights=self.fill_edge_weights)




class RadiusBipartite(GraphEdgeEncoderBase):
    r_cluster: float

    @beartype
    def __init__(self, r_cutoff: Union[Union[float, int], Sequence[Union[float, int, None]]],
                 irreps_sh: Optional[Union[str, o3.Irreps]],
                 length_enc_dim: Optional[int],
                 length_enc_type: Optional[str] = 'GaussianRadialBasis',
                 r_mincut_nonscalar_sh: Union[str, float, int, None] = 'default',  # Explicitly set this to ensure continuity of nonscalar spherical harmonics.
                 sh_cutoff: bool = False,
                 ):
        
        if isinstance(r_cutoff, int) or isinstance(r_cutoff, float):
            self.r_cluster = float(r_cutoff)
        elif isinstance(r_cutoff, Sequence) and not r_cutoff[-1] is None:
            self.r_cluster = float(r_cutoff[-1])
        else:
            raise TypeError(f"Wrong type for r_cutoff: {r_cutoff}")
        
        if length_enc_dim is None:
            length_enc = None
        else:
            if length_enc_type == 'SinusoidalPositionEmbeddings':
                length_enc = SinusoidalPositionEmbeddings(dim = length_enc_dim, 
                                                          max_val=self.r_cluster,
                                                          n=1000.)
            elif length_enc_type == 'BesselBasisEncoder':
                length_enc = torch.nn.Sequential(
                    BesselBasisEncoder(dim = 8,
                                       max_val = self.r_cluster,
                                       min_val = 0.,
                                       max_cutoff = True),
                    torch.nn.Linear(8, length_enc_dim)
                )
            elif length_enc_type == 'GaussianRadialBasis':
                length_enc = GaussianRadialBasis(dim = length_enc_dim, max_val=self.r_cluster)
            else:
                raise ValueError(f"Unknown length encoder type: {length_enc_type}")

        super().__init__(r_cutoff=r_cutoff, 
                         irreps_sh=irreps_sh,
                         r_mincut_nonscalar_sh=r_mincut_nonscalar_sh,
                         length_enc=length_enc,
                         sh_cutoff=sh_cutoff)

    def forward(self, src: FeaturedPoints, dst: FeaturedPoints, max_neighbors: int = 1000) -> GraphEdge:
        assert src.x.ndim == 2
        assert dst.x.ndim == 2
        edge = radius(x = src.x, y = dst.x, r=self.r_cluster, batch_x=src.b, batch_y=dst.b, max_num_neighbors=max_neighbors)
        edge_dst, edge_src = edge[0], edge[1]

        if not self.requires_encoding:
            return GraphEdge(edge_src=edge_src, edge_dst=edge_dst)
        
        return self._encode_edges(x_src=src.x, x_dst=dst.x, edge_src=edge_src, edge_dst=edge_dst)