from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3


from diffusion_edf import transforms
from diffusion_edf.equiformer.graph_attention_transformer import SeparableFCTP
from diffusion_edf.unet_feature_extractor import UnetFeatureExtractor
from diffusion_edf.multiscale_tensor_field import MultiscaleTensorField
from diffusion_edf.keypoint_extractor import KeypointExtractor, StaticKeypointModel
from diffusion_edf.gnn_data import FeaturedPoints, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffusion_edf.radial_func import SinusoidalPositionEmbeddings
from diffusion_edf.score_head import ScoreModelHead
from diffusion_edf.score_model_base import ScoreModelBase



class PointAttentiveScoreModel(ScoreModelBase):

    @beartype
    def __init__(self, 
                 query_model: str,
                 score_head_kwargs: Dict,
                 key_kwargs: Dict,
                 query_kwargs: Dict,
                 deterministic: bool = False):
        super().__init__()
        print("ScoreModel: Initializing Key Model")
        self.key_model = KeypointExtractor(
            **(key_kwargs),
            deterministic=deterministic
        )

        print("ScoreModel: Initializing Query Model")
        if query_model == 'KeypointExtractor':
            self.query_model = KeypointExtractor(
                **(query_kwargs),
                deterministic=deterministic
            )
        elif query_model == 'StaticKeypointModel':
            self.query_model = StaticKeypointModel(
                **(query_kwargs),
            )
        else:
            raise ValueError(f"Unknown query model: {query_model}")


        max_time: float = float(score_head_kwargs['max_time'])
        time_emb_mlp: List[int] = score_head_kwargs['time_emb_mlp']
        if 'lin_mult' in score_head_kwargs.keys():
            lin_mult: float = float(score_head_kwargs['lin_mult'])
        else:
            raise NotImplementedError()
            lin_mult: float = float(1.)
        if 'ang_mult' in score_head_kwargs.keys():
            ang_mult: float = float(score_head_kwargs['ang_mult'])
        else:
            raise NotImplementedError()
            ang_mult: float = math.sqrt(2.)
        edge_time_encoding: bool = score_head_kwargs['edge_time_encoding']
        query_time_encoding: bool = score_head_kwargs['query_time_encoding']

        key_tensor_field_kwargs = score_head_kwargs['key_tensor_field_kwargs']
        assert 'irreps_input' not in key_tensor_field_kwargs.keys()
        key_tensor_field_kwargs['irreps_input'] = self.key_model.irreps_output
        assert 'use_src_point_attn' not in key_tensor_field_kwargs.keys()
        key_tensor_field_kwargs['use_src_point_attn'] = True
        assert 'use_dst_point_attn' not in key_tensor_field_kwargs.keys()
        key_tensor_field_kwargs['use_dst_point_attn'] = False



        print("ScoreModel: Initializing Score Head")
        self.score_head = ScoreModelHead(max_time=max_time, 
                                         time_emb_mlp=time_emb_mlp,
                                         key_tensor_field_kwargs=key_tensor_field_kwargs,
                                         irreps_query_edf=self.query_model.irreps_output,
                                         lin_mult=lin_mult,
                                         ang_mult=ang_mult,
                                         edge_time_encoding=edge_time_encoding,
                                         query_time_encoding=query_time_encoding,
                                         )

        self.lin_mult = self.score_head.lin_mult
        self.ang_mult = self.score_head.ang_mult

    def get_key_pcd_multiscale(self, pcd: FeaturedPoints) -> List[FeaturedPoints]:
        return [self.key_model(pcd)]
    
    def get_query_pcd(self, pcd: FeaturedPoints) -> FeaturedPoints:
        return self.query_model(pcd)