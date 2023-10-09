from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict, Sequence
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3


from diffusion_edf import transforms
from diffusion_edf.equiformer.graph_attention_transformer import SeparableFCTP
from diffusion_edf.unet_feature_extractor import UnetFeatureExtractor
from diffusion_edf.forward_only_feature_extractor import ForwardOnlyFeatureExtractor
from diffusion_edf.multiscale_tensor_field import MultiscaleTensorField
from diffusion_edf.keypoint_extractor import KeypointExtractor, StaticKeypointModel
from diffusion_edf.gnn_data import FeaturedPoints, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffusion_edf.radial_func import SinusoidalPositionEmbeddings
from diffusion_edf.score_head import ScoreModelHead


class ScoreModelBase(torch.nn.Module):
    lin_mult: float
    ang_mult: float
    q_indices: torch.Tensor
    q_factor: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long), persistent=False)
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]), persistent=False)

    def get_key_pcd_multiscale(self, pcd: FeaturedPoints) -> List[FeaturedPoints]:
        raise NotImplementedError
    
    def get_query_pcd(self, pcd: FeaturedPoints) -> FeaturedPoints:
        raise NotImplementedError

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

        key_pcd_multiscale: List[FeaturedPoints] = self.get_key_pcd_multiscale(key_pcd)
        query_pcd: FeaturedPoints = self.get_query_pcd(query_pcd)

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

        # ang_loss = ang_loss * ((self.lin_mult/self.ang_mult)**2)
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
            #"key_fp": detach_featured_points(key_pcd_multiscale[0]),
            "key_fp": None,
            "query_fp": detach_featured_points(query_pcd),
        }

        tensor_info: Dict[str, torch.Tensor] = {
            'ang_score': ang_score.detach(),
            'lin_score': lin_score.detach(),
        }

        return loss, fp_info, tensor_info, statistics

    @torch.jit.export
    def sample(self, T_seed: torch.Tensor,
               scene_pcd_multiscale: List[FeaturedPoints], 
               grasp_pcd: FeaturedPoints,
               diffusion_schedules: List[
                                        Union[List[float], Tuple[float, float]]
                                    ],
               N_steps: List[int], 
               timesteps: List[float],
               temperatures: Union[Union[int, float], Sequence[Union[int, float]]] = 1.0,
               log_t_schedule: bool = True,
               time_exponent_temp: float = 0.5, # Theoretically, this should be zero.
               time_exponent_alpha: float = 0.5, # Most commonly used exponent in image generation is 1.0, but it is too slow in our case.
               ) -> torch.Tensor:
        """
        alpha = timestep * L^2 * (t^time_exponent_alpha)
        T = temperature * (t^time_exponent_temp)
        """

        if isinstance(temperatures, (int, float)):
            temperatures = [float(temperatures) for _ in range(len(diffusion_schedules))]
        
        # ---------------------------------------------------------------------------- #
        # Convert Data Type
        # ---------------------------------------------------------------------------- #
        device = T_seed.device
        dtype = T_seed.dtype
        T = T_seed.clone().detach().type(torch.float64)
        temperatures = torch.tensor(temperatures, device=device, dtype=torch.float64)
        diffusion_schedules = torch.tensor(diffusion_schedules, device=device, dtype=torch.float64)
        

        # ---------------------------------------------------------------------------- #
        # Begin Loop
        # ---------------------------------------------------------------------------- #
        Ts = [T.clone().detach()]
        steps = 0
        for n, schedule in enumerate(diffusion_schedules):
            temperature_base = temperatures[n]
            if log_t_schedule:
                t_schedule = torch.logspace(
                    start=torch.log(schedule[0]), 
                    end=torch.log(schedule[1]), 
                    steps=N_steps[n], 
                    base=torch.e, 
                    device=device, 
                    dtype=torch.float64,
                ).unsqueeze(-1)
            else:
                t_schedule = torch.linspace(
                    start=schedule[0], 
                    end=schedule[1], 
                    steps=N_steps[n], 
                    device=device, 
                    dtype=torch.float64
                ).unsqueeze(-1)

            print(f"{self.__class__.__name__}: sampling with (temp_base: {temperature_base} || t_schedule: {schedule.detach().cpu().numpy()})")
            for i in tqdm(range(len(t_schedule))):
                t = t_schedule[i]
                temperature = temperature_base * torch.pow(t,time_exponent_temp)
                alpha_ang = (self.ang_mult **2) * torch.pow(t,time_exponent_alpha) * timesteps[n]
                alpha_lin = (self.lin_mult **2) * torch.pow(t,time_exponent_alpha) * timesteps[n]

                with torch.no_grad():
                    (ang_score_dimless, lin_score_dimless) = self.score_head(Ts=T.view(-1,7).type(dtype), 
                                                                            key_pcd_multiscale=scene_pcd_multiscale,
                                                                            query_pcd=grasp_pcd,
                                                                            time = t.repeat(len(T)).type(dtype))
                ang_score = ang_score_dimless.type(torch.float64) / (self.ang_mult * torch.sqrt(t))
                lin_score = lin_score_dimless.type(torch.float64) / (self.lin_mult * torch.sqrt(t))

                ang_noise = torch.sqrt(temperature*alpha_ang) * torch.randn_like(ang_score, dtype=torch.float64) 
                lin_noise = torch.sqrt(temperature*alpha_lin) * torch.randn_like(lin_score, dtype=torch.float64)

                ang_disp = (alpha_ang/2) * ang_score + ang_noise
                lin_disp = (alpha_lin/2) * lin_score + lin_noise


                L = T.detach()[...,self.q_indices] * (self.q_factor.type(torch.float64))
                q, x = T[...,:4], T[...,4:]
                dq = torch.einsum('...ij,...j->...i', L, ang_disp)
                dx = transforms.quaternion_apply(q, lin_disp)
                q = transforms.normalize_quaternion(q + dq)
                T = torch.cat([q, x+dx], dim=-1)

                # dT = transforms.se3_exp_map(torch.cat([lin_disp, ang_disp], dim=-1))
                # dT = torch.cat([transforms.matrix_to_quaternion(dT[..., :3, :3]), dT[..., :3, 3]], dim=-1)
                # T = transforms.multiply_se3(T, dT)
                steps += 1
                Ts.append(T.clone().detach())

        Ts.append(T.clone().detach())
        Ts = torch.stack(Ts, dim=0).detach()

        return Ts

    def forward(self, Ts: torch.Tensor, 
                time: torch.Tensor, 
                key_pcd: FeaturedPoints, 
                query_pcd: FeaturedPoints, 
                debug: bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                              Optional[Tuple[List[FeaturedPoints], FeaturedPoints]]]:

        key_pcd_multiscale: List[FeaturedPoints] = self.get_key_pcd_multiscale(key_pcd)
        query_pcd: FeaturedPoints = self.get_query_pcd(query_pcd)

        score: Tuple[torch.Tensor, torch.Tensor] = self.score_head(Ts = Ts, 
                                                                   key_pcd_multiscale = key_pcd_multiscale, 
                                                                   query_pcd = query_pcd,
                                                                   time = time)
        if debug:
            debug_output = ([detach_featured_points(key_pcd) for key_pcd in key_pcd_multiscale], detach_featured_points(query_pcd))
        else:
            debug_output = None
                                                                   
        return score, debug_output
