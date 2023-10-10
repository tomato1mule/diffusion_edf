from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from torch.nn import functional as F
from e3nn import o3
from torch_cluster import fps


from diffusion_edf import transforms
from diffusion_edf.equiformer.graph_attention_transformer import SeparableFCTP
from diffusion_edf.unet_feature_extractor import UnetFeatureExtractor
from diffusion_edf.forward_only_feature_extractor import ForwardOnlyFeatureExtractor
from diffusion_edf.multiscale_tensor_field import MultiscaleTensorField
from diffusion_edf.gnn_data import FeaturedPoints, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffusion_edf.radial_func import SinusoidalPositionEmbeddings


class StaticKeypointModel(torch.nn.Module):
    @beartype
    def __init__(self, 
                 keypoint_coords: Union[torch.Tensor, List],
                 irreps_output: Union[o3.Irreps, str]):
        
        super().__init__()
        keypoint_coords = torch.tensor(keypoint_coords)
        assert keypoint_coords.ndim == 2 and keypoint_coords.shape[-1] == 3, f"{keypoint_coords.shape}"  # (nPoints, 3)
        self.irreps_output = o3.Irreps(irreps_output)

        self.register_buffer("keypoint_coords", keypoint_coords)
        self.keypoint_features = torch.nn.Parameter(self.irreps_output.randn(len(self.keypoint_coords), -1))
        self.keypoint_weights = torch.nn.Parameter(torch.randn(len(self.keypoint_coords)))

    def forward(self, input_points: FeaturedPoints) -> FeaturedPoints:
        b = input_points.b
        assert b.ndim == 1
        batch_unique = torch.unique(b)
        x = self.keypoint_coords.repeat(len(batch_unique), 1)
        f = self.keypoint_features.repeat(len(batch_unique), 1)
        w = torch.sigmoid(self.keypoint_weights)
        w = w.repeat(len(batch_unique))
        b = batch_unique.repeat(len(self.keypoint_coords))
        
        return FeaturedPoints(x=x, f=f, b=b, w=w)


class KeypointExtractor(torch.nn.Module):
    weight_pre_emb_dim: int
    deterministic: bool
    pool_ratio: float

    @beartype
    def __init__(self, 
                 feature_extractor_kwargs: Dict,
                 tensor_field_kwargs: Dict,
                 keypoint_kwargs: Dict,
                 feature_extractor_name: str = 'UnetFeatureExtractor', # ForwardOnlyFeatureExtractor
                 weight_activation: str = 'sigmoid',
                 weight_mult: Optional[Union[float, int]] = None,
                 deterministic: bool = False,):
        super().__init__()
        self.deterministic = deterministic
        self.pool_ratio = float(keypoint_kwargs['pool_ratio'])
        self.keypoint_bbox: Optional[List[List[float]]] = keypoint_kwargs.get('bbox', None)
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
        if weight_mult is None:
            self.weight_mult_logit = None
        else:
            self.weight_mult_logit = torch.nn.Parameter(
                torch.log(torch.exp(torch.tensor(float(weight_mult)))-1) # inverse softplus
            )

        #### Feature Extractor ####
        if feature_extractor_name == 'UnetFeatureExtractor':
            self.feature_extractor = UnetFeatureExtractor(
                **(feature_extractor_kwargs),
                deterministic=self.deterministic
            )
        elif feature_extractor_name == 'ForwardOnlyFeatureExtractor':
            self.feature_extractor = ForwardOnlyFeatureExtractor(
                **(feature_extractor_kwargs),
                deterministic=self.deterministic
            )
        else:
            raise ValueError(f"Unknown feature extractor name: {feature_extractor_name}")


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
            torch.nn.Sigmoid() if weight_activation == 'sigmoid' else torch.nn.Identity(),
        )
        if weight_activation == 'sigmoid' or 'none':
            self.weight_activation = None
        elif weight_activation == 'softmax':
            self.weight_activation = torch.nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unknown weight activation: {weight_activation}")

        self.irreps_output = o3.Irreps(self.tensor_field.irreps_output)

    def init_query_points(self, src_points: FeaturedPoints, 
                       retain_feature: bool = False,
                       retain_weight: bool = False) -> FeaturedPoints:
        assert src_points.x.ndim == 2 and src_points.x.shape[-1] == 3, f"{src_points.x.shape}"
        x = src_points.x
        f = src_points.f
        b = src_points.b
        w = src_points.w
        
        
        if self.keypoint_bbox is not None:
            keypoint_bbox = torch.tensor(self.keypoint_bbox, dtype=x.dtype, device=x.device)
            inrange_idx = ((x >= keypoint_bbox[:,0]) * (x <= keypoint_bbox[:,1])).all(dim=-1).nonzero().squeeze()
            x = x.index_select(index=inrange_idx, dim=0)
            f = f.index_select(index=inrange_idx, dim=0)
            b = b.index_select(index=inrange_idx, dim=0)
            if w is not None:
                w = w.index_select(index=inrange_idx, dim=0)

        node_dst_idx = fps(src=x.detach(), 
                           batch=b.detach(), 
                           ratio=self.pool_ratio, 
                           random_start=not self.deterministic)
        

        if retain_feature:
            x = x.index_select(index=node_dst_idx, dim=0)
            b = b.index_select(index=node_dst_idx, dim=0)
            f = f.index_select(index=node_dst_idx, dim=0)
        else:
            x = x.index_select(index=node_dst_idx, dim=0)
            b = b.index_select(index=node_dst_idx, dim=0)
            f = torch.empty_like(x)
            
        if retain_weight and w is not None:
            w = w.index_select(index=node_dst_idx, dim=0)
        else:
            w = None

        return FeaturedPoints(x=x, f=f, b=b, w=w)
    
    def get_query_points(self, src_points: FeaturedPoints) -> FeaturedPoints:
        # DONT FORGET self.keypoint_bbox if you'd like to add sth here
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
        if self.weight_activation is not None:
            weights = self.weight_activation(weights)
        if self.weight_mult_logit is not None:
            weights = weights * F.softplus(self.weight_mult_logit)

        output_points: FeaturedPoints = set_featured_points_attribute(points = output_points, w=weights)
        return output_points



