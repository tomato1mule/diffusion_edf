from typing import List, Optional, Union, Tuple, Dict
import warnings
import math
from beartype import beartype

import torch
from e3nn import o3


from diffusion_edf.gnn_block import EquiformerBlock
from diffusion_edf.utils import multiply_irreps
from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge, set_graph_edge_attribute, cat_graph_edges, cat_featured_points
from diffusion_edf.graph_parser import RadiusBipartite, InfiniteBipartite


class MultiscaleTensorField(torch.nn.Module):
    r_cluster_multiscale: List[Optional[float]]
    n_scales: int
    cutoff_method: str

    @beartype
    def __init__(self,
        irreps_input: Union[str, o3.Irreps], 
        irreps_output: Union[str, o3.Irreps], 
        irreps_sh: Union[str, o3.Irreps], 
        num_heads: int, 
        fc_neurons: List[int],
        length_emb_dim: int,
        irreps_query: Optional[Union[str, o3.Irreps]],      # Explicitly set to None if not want to use query features
        r_cluster_multiscale: List[Optional[float]],
        edge_context_emb_dim: Optional[int],                # Encode context to edge scalars. Set to None if not want to encode context to the edges      
        r_mincut_nonscalar_sh: Optional[float] = None,
        length_enc_max_r: Optional[float] = None,
        n_scales: Optional[int] = None,                     # This parameter is just for double checking. If set to None, it will be automatically set to r_cluster_multiscale.
        n_layers: int = 1,
        irreps_mlp_mid: Union[str, o3.Irreps, int] = 3,
        attn_type: str = 'mlp',
        alpha_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0,
        use_src_point_attn: bool = False,
        use_dst_point_attn: bool = False,
        cutoff_method: str = 'edge_attn'):

        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_sh = o3.Irreps(irreps_sh)
        if irreps_query is not None:
            self.use_dst_feature = True
            self.irreps_query = o3.Irreps(irreps_query)
        else:
            self.use_dst_feature = False
            self.irreps_query = None
        self.num_heads: int = num_heads
        self.fc_neurons: List[int] = fc_neurons
        

        self.length_emb_dim = length_emb_dim
        assert self.length_emb_dim > 0, f"{self.length_emb_dim}"
        self.context_emb_dim = edge_context_emb_dim

        if self.fc_neurons[0] == -1:
            if self.context_emb_dim is None:
                self.fc_neurons[0] = self.length_emb_dim
            else:
                self.fc_neurons[0] = self.length_emb_dim + self.context_emb_dim

        if self.context_emb_dim is None:
            assert fc_neurons[0] == self.length_emb_dim, f"{fc_neurons[0]} != {self.length_emb_dim}"
        else:
            assert self.context_emb_dim > 0, f"{self.context_emb_dim}"
            assert fc_neurons[0] == self.length_emb_dim + self.context_emb_dim, f"{fc_neurons[0]} != {self.length_emb_dim} + {self.context_emb_dim}"


        self.cutoff_method = cutoff_method
        if self.cutoff_method == 'edge_attn': # It has better continuity, but slow.
            use_edge_weights = True
            sh_cutoff = False
        elif self.cutoff_method == 'sh':      # It has worse continuity, but fast.
            use_edge_weights = False
            sh_cutoff = True
        else:
            raise ValueError(f"Unknown cutoff method: {cutoff_method}")
        


        self.r_cluster_multiscale = r_cluster_multiscale
        if n_scales is None:
            self.n_scales = len(self.r_cluster_multiscale)
        else:
            assert n_scales == len(self.r_cluster_multiscale)
            self.n_scales = n_scales
        if r_mincut_nonscalar_sh is None:
            assert self.r_cluster_multiscale[0] is not None
            r_mincut_nonscalar_sh = 0.01 * self.r_cluster_multiscale[0]
        if length_enc_max_r is None:
            assert self.r_cluster_multiscale[-1] is not None
        else:
            assert self.r_cluster_multiscale[-1] is None, "You don't need to provide length_enc_max_r"

        # !!!!!!!!!!!!!!TODO!!!!!!!!!!!! 
        # min_offset = self.r_cluster_multiscale[0] * 0.01
        # self.mincut_offsets = [min_offset] + [max(min_offset, offset - 0.2*(cutoff - offset)) for offset, cutoff in zip(self.r_cluster_multiscale[:-1], self.r_cluster_multiscale[1:])],

        self.graph_parsers = torch.nn.ModuleList()
        self.edge_scalars_pre_linears = torch.nn.ModuleList()
        
        fill_edge_weights = False
        infinite = False
        for n in range(self.n_scales):
            r_cutoff = self.r_cluster_multiscale[n]
            if r_cutoff is None:
                self.graph_parsers.append(
                    InfiniteBipartite(
                        length_enc_max_r=length_enc_max_r,
                        irreps_sh=self.irreps_sh,
                        length_enc_dim=self.length_emb_dim,
                        length_enc_type='SinusoidalPositionEmbeddings',
                        r_mincut_nonscalar_sh=r_mincut_nonscalar_sh,
                        sh_cutoff = sh_cutoff,
                        fill_edge_weights=fill_edge_weights
                    )
                )
                infinite = True
            else:
                assert not infinite, f"Finite cluster radius cannot come after infinite cluster radius, {self.r_cluster_multiscale}"                
                self.graph_parsers.append(
                    RadiusBipartite(
                        # r_cutoff=[self.mincut_offsets[n], ??, ?? , self.r_cluster_multiscale[n]],
                        r_cutoff=self.r_cluster_multiscale[n],
                        irreps_sh=self.irreps_sh,
                        length_enc_dim=self.length_emb_dim,
                        length_enc_type='GaussianRadialBasis',
                        r_mincut_nonscalar_sh=r_mincut_nonscalar_sh,
                        sh_cutoff = sh_cutoff
                    )
                )
                if use_edge_weights:
                    fill_edge_weights = True
            self.edge_scalars_pre_linears.append(
                torch.nn.Sequential(
                    torch.nn.Linear(fc_neurons[0], fc_neurons[0]),
                    torch.nn.SiLU(inplace=True)
                )
            )
        
        self.n_layers = n_layers
        assert self.n_layers >= 1
        self.gnn_block_init = EquiformerBlock(irreps_src = self.irreps_input, 
                                              irreps_dst = self.irreps_query if self.use_dst_feature else self.irreps_input, 
                                              irreps_emb = self.irreps_input, 
                                              irreps_output = self.irreps_output if self.n_layers == 1 else self.irreps_input, 
                                              irreps_edge_attr = self.irreps_sh, 
                                              num_heads = self.num_heads, 
                                              fc_neurons = fc_neurons,
                                              irreps_mlp_mid = irreps_mlp_mid,
                                              attn_type = attn_type,
                                              alpha_drop = alpha_drop,
                                              proj_drop = proj_drop,
                                              drop_path_rate = drop_path_rate,
                                              use_dst_feature = self.use_dst_feature,
                                              skip_connection = True,
                                              bias = True,
                                              use_src_point_attn=use_src_point_attn,
                                              use_dst_point_attn=use_dst_point_attn,
                                              use_edge_weights=use_edge_weights)
        
        self.gnn_blocks = torch.nn.ModuleList()
        for n in range(self.n_layers-1):
            self.gnn_blocks.append(
                EquiformerBlock(irreps_src = self.irreps_input, 
                                irreps_dst = self.irreps_input, 
                                irreps_emb = self.irreps_input, 
                                irreps_output = self.irreps_output if n == self.n_layers-1 else self.irreps_input, 
                                irreps_edge_attr = self.irreps_sh, 
                                num_heads = self.num_heads, 
                                fc_neurons = fc_neurons,
                                irreps_mlp_mid = irreps_mlp_mid,
                                attn_type = attn_type,
                                alpha_drop = alpha_drop,
                                proj_drop = proj_drop,
                                drop_path_rate = drop_path_rate,
                                use_dst_feature = True,
                                skip_connection = True,
                                bias = True,
                                use_src_point_attn=use_src_point_attn,
                                use_dst_point_attn=use_dst_point_attn,
                                use_edge_weights=use_edge_weights)
            )
        
    def forward(self, query_points: FeaturedPoints,
                input_points_multiscale: List[FeaturedPoints],
                context_emb: Optional[List[torch.Tensor]] = None,
                max_neighbors: int = 1000) -> FeaturedPoints:
        assert len(input_points_multiscale) == self.n_scales
        assert query_points.x.ndim == 2 # (Nq, 3)
        if self.context_emb_dim is not None:
            edge_encode_context = True
            assert context_emb is not None
        else:
            edge_encode_context = False
            assert context_emb is None

        n_total_points: int = 0
        graph_edges_flattend: Optional[GraphEdge] = None
        input_points_flattend: Optional[FeaturedPoints] = None
        for n, (graph_parser, edge_scalars_pre_linear) in enumerate(zip(self.graph_parsers, self.edge_scalars_pre_linears)):
            input_points: FeaturedPoints = input_points_multiscale[n]
            assert input_points.x.ndim == 2 and input_points.x.shape[-1] == 3, f"{input_points.x.shape}"

            ### Parse Graph ###
            graph_edge: GraphEdge = graph_parser(src=input_points, dst=query_points, max_neighbors=max_neighbors)

            ### Encode length and context embeddings ###
            edge_scalars = graph_edge.edge_scalars
            assert isinstance(edge_scalars, torch.Tensor) # to tell torch.jit.script that it is a tensor
            
            if edge_encode_context is True:
                # assert isinstance(context_emb, torch.Tensor)
                # assert context_emb.ndim == 2  # (Nq, tEmb)
                # assert self.context_emb_dim == context_emb.shape[-1], f"{self.context_emb_dim} != {context_emb.shape[-1]} of {context_emb.shape}"
                # context_emb = context_emb.index_select(0, graph_edge.edge_dst)            # (nEdge, cEmb)
                # edge_scalars = torch.cat([edge_scalars, context_emb], dim=-1)  # (nEdge, Emb = lEmb + cEmb)
                edge_scalars = torch.cat([
                    edge_scalars, 
                    context_emb[n].index_select(0, graph_edge.edge_dst)
                ], dim=-1) # (nEdge, Emb = lEmb + cEmb)
                # edge_scalars = edge_scalars.type(torch.float32) # To avoid JIT type bug
                edge_scalars = edge_scalars.type(edge_scalars_pre_linear[0].weight.dtype) # To avoid JIT type bug
            else:
                pass                                  # (nEdge, Emb = lEmb)
            
            edge_scalars = edge_scalars_pre_linear(edge_scalars)                          # (nEdge, Emb)
            ### Flatten graph ###
            graph_edge = set_graph_edge_attribute(graph_edge=graph_edge, 
                                                  edge_scalars=edge_scalars, 
                                                  edge_src = graph_edge.edge_src + n_total_points)
            n_total_points = n_total_points + len(input_points.x)
            if n == 0:
                assert graph_edges_flattend is None and input_points_flattend is None
                graph_edges_flattend = graph_edge
                input_points_flattend = input_points
            else:
                assert graph_edges_flattend is not None and input_points_flattend is not None
                graph_edges_flattend = cat_graph_edges(graph_edges_flattend, graph_edge)
                input_points_flattend = cat_featured_points(input_points_flattend, input_points)

        if len(graph_edges_flattend.edge_src) == 0:
            warnings.warn("Multiscale Tensor Field: zero edges detected!")

        output_points: FeaturedPoints = self.gnn_block_init(src_points=input_points_flattend,
                                                            dst_points=query_points,
                                                            graph_edge=graph_edges_flattend)
        for block in self.gnn_blocks:
            output_points: FeaturedPoints = block(src_points=input_points_flattend,
                                                  dst_points=output_points,
                                                  graph_edge=graph_edges_flattend)
        
        return output_points
        