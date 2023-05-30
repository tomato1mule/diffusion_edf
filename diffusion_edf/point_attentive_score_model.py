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



class PointAttentiveScoreModel(torch.nn.Module):

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

        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long), persistent=False)
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]), persistent=False)

    @torch.jit.ignore()
    def to(self, *args, **kwargs):
        for module in self.children():
            if isinstance(module, torch.nn.Module):
                module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @torch.jit.export
    def get_train_loss(self, Ts: torch.Tensor, 
                       time: torch.Tensor, 
                       key_pcd: FeaturedPoints, 
                       query_pcd: FeaturedPoints, 
                       target_ang_score: torch.Tensor,
                       target_lin_score: torch.Tensor,
                       ) -> Tuple[torch.Tensor, 
                                  Dict[str, Optional[FeaturedPoints]], 
                                  Dict[str, torch.Tensor], 
                                  Dict[str, torch.Tensor]]:
        assert target_ang_score.ndim == 2 and target_ang_score.shape[-1] == 3, f"{target_ang_score.shape}"
        assert target_lin_score.ndim == 2 and target_lin_score.shape[-1] == 3, f"{target_lin_score.shape}"
        assert time.ndim == 1 and target_ang_score.shape[-1] == 3, f"{target_ang_score.shape}"
        assert len(time) == len(target_ang_score) == len(target_lin_score)

        #key_pcd_multiscale: List[FeaturedPoints] = self.key_feature_extractor(key_pcd)
        key_pcd_multiscale: List[FeaturedPoints] = [self.key_model(key_pcd)]
        query_pcd: FeaturedPoints = self.query_model(query_pcd)

        ang_score, lin_score = self.score_head(Ts = Ts, 
                                               key_pcd_multiscale = key_pcd_multiscale, 
                                               query_pcd = query_pcd,
                                               time = time)
        
        target_ang_score = target_ang_score * torch.sqrt(time[..., None]) * self.ang_mult
        target_lin_score = target_lin_score * torch.sqrt(time[..., None]) * self.lin_mult
        ang_score_diff = target_ang_score - ang_score
        lin_score_diff = target_lin_score - lin_score
        ang_loss = torch.sum(torch.square(ang_score_diff), dim=-1).mean(dim=-1)
        lin_loss = torch.sum(torch.square(lin_score_diff), dim=-1).mean(dim=-1)

        ang_loss = ang_loss * ((self.lin_mult/self.ang_mult)**2)
        loss = ang_loss + lin_loss


        target_norm_ang, target_norm_lin = torch.norm(target_ang_score.detach(), dim=-1), torch.norm(target_lin_score.detach(), dim=-1) # Shape: (nT, ), (nT, )
        score_norm_ang, score_norm_lin = torch.norm(ang_score.detach(), dim=-1), torch.norm(lin_score.detach(), dim=-1)         # Shape: (nT, ), (nT, )
        dp_align_ang = torch.einsum('...i,...i->...', ang_score.detach(), target_ang_score.detach()) # Shape: (nT, )
        dp_align_lin = torch.einsum('...i,...i->...', lin_score.detach(), target_lin_score.detach()) # Shape: (nT, )
        dp_align_ang_normalized = dp_align_ang / target_norm_ang / score_norm_ang # Shape: (nT, )
        dp_align_lin_normalized = dp_align_lin / target_norm_lin / score_norm_lin # Shape: (nT, )

        statistics: Dict[str, torch.Tensor] = {
            "Loss/train": loss.item(),
            "Loss/angular": ang_loss.item(),
            "Loss/linear": lin_loss.item(),
            "norm/target_ang": target_norm_ang.mean(dim=-1).item(),
            "norm/target_lin": target_norm_lin.mean(dim=-1).item(),
            "norm/inferred_ang": score_norm_ang.mean(dim=-1).item(),
            "norm/inferred_lin": score_norm_lin.mean(dim=-1).item(),
            "alignment/unnormalized/ang": dp_align_ang.mean(dim=-1).item(),
            "alignment/unnormalized/lin": dp_align_lin.mean(dim=-1).item(),
            "alignment/normalized/ang": dp_align_ang_normalized.mean(dim=-1).item(),
            "alignment/normalized/lin": dp_align_lin_normalized.mean(dim=-1).item(),
        }

        fp_info: Dict[str, Optional[FeaturedPoints]] = {
            "key_fp": detach_featured_points(key_pcd_multiscale[0]),
            #"key_fp": None,
            "query_fp": detach_featured_points(query_pcd),
        }

        tensor_info: Dict[str, torch.Tensor] = {
            'ang_score': ang_score.detach(),
            'lin_score': lin_score.detach(),
        }

        return loss, fp_info, tensor_info, statistics

        

    def forward(self, Ts: torch.Tensor, 
                time: torch.Tensor, 
                key_pcd: FeaturedPoints, 
                query_pcd: FeaturedPoints, 
                debug: bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                              Optional[Tuple[List[FeaturedPoints], FeaturedPoints]]]:

        #key_pcd_multiscale: List[FeaturedPoints] = self.key_feature_extractor(key_pcd)
        key_pcd_multiscale: List[FeaturedPoints] = [self.key_model(key_pcd)]
        query_pcd: FeaturedPoints = self.query_model(query_pcd)

        score: Tuple[torch.Tensor, torch.Tensor] = self.score_head(Ts = Ts, 
                                                                   key_pcd_multiscale = key_pcd_multiscale, 
                                                                   query_pcd = query_pcd,
                                                                   time = time)
        if debug:
            debug_output = ([detach_featured_points(key_pcd) for key_pcd in key_pcd_multiscale], detach_featured_points(query_pcd))
        else:
            debug_output = None
                                                                   
        return score, debug_output
