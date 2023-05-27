from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3
from torch_cluster import fps


from diffusion_edf import transforms
from diffusion_edf.equiformer.graph_attention_transformer import SeparableFCTP
from diffusion_edf.feature_extractor import UnetFeatureExtractor
from diffusion_edf.multiscale_tensor_field import MultiscaleTensorField
from diffusion_edf.gnn_data import FeaturedPoints, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffusion_edf.radial_func import SinusoidalPositionEmbeddings



class KeypointExtractor(torch.nn.Module):
    weight_pre_emb_dim: int
    deterministic: bool
    pool_ratio: float

    @beartype
    def __init__(self, 
                 feature_extractor_kwargs: Dict,
                 tensor_field_kwargs: Dict,
                 keypoint_kwargs: Dict,
                 deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic
        self.pool_ratio = float(keypoint_kwargs['pool_ratio'])
        weight_pre_emb_dim: Optional[int] = keypoint_kwargs['weight_pre_emb_dim']
        if weight_pre_emb_dim:
            pass
        else:
            weight_pre_emb_dim = 0
            for n, (l,p) in self.feature_extractor.irreps_output:
                if p != 1:
                    raise NotImplementedError(f"Only SE(3) equivariance is implemented! (irreps_output = {self.feature_extractor.irreps_output}")
                if l == 0:
                    weight_pre_emb_dim += n
        self.weight_pre_emb_dim = weight_pre_emb_dim
        assert self.weight_pre_emb_dim, f"{self.weight_pre_emb_dim}"


        #### Feature Extractor ####
        self.feature_extractor = UnetFeatureExtractor(
            **(feature_extractor_kwargs),
            deterministic=self.deterministic
        )


        #### Feature EDF ####
        assert 'irreps_input' not in tensor_field_kwargs.keys()
        tensor_field_kwargs['irreps_input'] = feature_extractor_kwargs['irreps_output']

        assert 'irreps_query' not in tensor_field_kwargs.keys()
        tensor_field_kwargs['irreps_query'] = None

        assert 'edge_context_emb_dim' not in tensor_field_kwargs.keys()
        tensor_field_kwargs['edge_context_emb_dim'] = None

        self.tensor_field = MultiscaleTensorField(**(tensor_field_kwargs))


        #### Equivariant Weight Field ####
        tensor_field_kwargs['irreps_output'] = o3.Irreps(f"{weight_pre_emb_dim}x0e")
        self.weight_field = MultiscaleTensorField(**(tensor_field_kwargs))
        self.weight_post = torch.nn.Sequential(
            torch.nn.LayerNorm(self.weight_pre_emb_dim),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(self.weight_pre_emb_dim, 1),
            torch.nn.Sigmoid(),
        )

        self.irreps_output = o3.Irreps(self.tensor_field.irreps_output)

    def init_query_points(self, src_points: FeaturedPoints, 
                       retain_feature: bool = False,
                       retain_weight: bool = False) -> FeaturedPoints:
        assert src_points.x.ndim == 2 and src_points.x.shape[-1] == 3, f"{src_points.x.shape}"

        node_dst_idx = fps(src=src_points.x.detach(), 
                           batch=src_points.b.detach(), 
                           ratio=self.pool_ratio, 
                           random_start=not self.deterministic)
        

        if retain_feature:
            x = src_points.x.index_select(index=node_dst_idx, dim=0)
            b = src_points.b.index_select(index=node_dst_idx, dim=0)
            f = src_points.f.index_select(index=node_dst_idx, dim=0)
        else:
            x = src_points.x.index_select(index=node_dst_idx, dim=0)
            b = src_points.b.index_select(index=node_dst_idx, dim=0)
            f = torch.empty_like(x)
        if retain_weight:
            if src_points.w is None:
                w = None
            else:
                w = src_points.w.index_select(index=node_dst_idx, dim=0)
        else:
            w = None

        return FeaturedPoints(x=x, f=f, b=b, w=w)
    
    def get_query_points(self, src_points: FeaturedPoints) -> FeaturedPoints:
        return self.init_query_points(src_points=src_points, retain_feature=False, retain_weight=False)

    def forward(self, input_points: FeaturedPoints, max_neighbors: Optional[int] = 1000) -> FeaturedPoints:
        output_points_multiscale: List[FeaturedPoints] = self.feature_extractor(input_points)
        query_points = self.get_query_points(src_points=input_points)
        output_points = self.tensor_field(query_points=query_points,
                                          input_points_multiscale = output_points_multiscale,
                                          context_emb = None,
                                          max_neighbors = max_neighbors) # Features: (nQ, F)
        
        weights = self.weight_field(query_points=query_points,
                                    input_points_multiscale = output_points_multiscale,
                                    context_emb = None,
                                    max_neighbors = max_neighbors).f # Features: (nQ, wEmb)
        weights = self.weight_post(weights).squeeze(-1) # Features: (nQ, )

        output_points: FeaturedPoints = set_featured_points_attribute(points = output_points, w=weights)
        return output_points



