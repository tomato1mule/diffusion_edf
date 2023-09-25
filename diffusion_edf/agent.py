import os
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
from typing import List, Tuple, Optional, Union, Iterable, Dict, Sequence
import math
import argparse
import warnings

from beartype import beartype
import yaml
import torch

from edf_interface.data import SE3, PointCloud, TargetPoseDemo
from diffusion_edf.gnn_data import FeaturedPoints, pcd_to_featured_points
from diffusion_edf.trainer import DiffusionEdfTrainer
from diffusion_edf import train_utils

torch.set_printoptions(precision=4, sci_mode=False)


@beartype
def get_models(configs_root_dir: str, 
               train_configs_file: str, 
               task_configs_file: str, 
               checkpoint_dir: str,
               device: str,
               n_warmups: int = 10,
               compile_score_head: bool = False,
               strict_load: bool = False,
               half_precision: bool = False,
               ):

    trainer = DiffusionEdfTrainer(
        configs_root_dir=configs_root_dir,
        train_configs_file=train_configs_file,
        task_configs_file=task_configs_file,
        device=device
    )

    trainer._init_dataloaders(half_precision=half_precision)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The TorchScript type system doesn*')
        
        model = trainer.get_model(
            checkpoint_dir=checkpoint_dir,
            deterministic=False, 
            device = device,
            strict=strict_load
        ).eval()
        model.diffusion_schedules = trainer.diffusion_schedules
    if half_precision:
        model = model.half()
        
    if compile_score_head:
        model.score_head = torch.jit.script(model.score_head)

    print(f"Warming up the model for {n_warmups} iterations", flush=True)
    if n_warmups:
        trainer.warmup_score_model(
            score_model = model, 
            n_warmups=n_warmups
        )
    
    return model

@beartype
class DiffusionEdfAgent():
    task_type: str
    def __init__(self, model_kwargs_list: List[Dict], 
                 preprocess_config,
                 unprocess_config,
                 device: str,
                 compile_score_head: bool = False,
                 half_precision: bool = False,):
        self.models = []
        for kwargs in model_kwargs_list:
            self.models.append(get_models(**kwargs, device=device, compile_score_head=compile_score_head, half_precision=half_precision))

        self.proc_fn = train_utils.compose_proc_fn(preprocess_config=preprocess_config)
        self.unprocess_fn = train_utils.compose_proc_fn(preprocess_config=unprocess_config)

    def sample(self, scene_pcd: PointCloud, 
               grasp_pcd: PointCloud, 
               Ts_init: SE3,
               N_steps_list: List[List[int]],
               timesteps_list: List[List[float]],
               temperatures_list: List[Union[Union[int, float], Sequence[Union[int, float]]]],
               diffusion_schedules_list: Optional[List[Optional[List[Union[List[float], Tuple[float, float]]]]]] = None,
               log_t_schedule: bool = True,
               time_exponent_temp: float = 1.0, # Theoretically, this should be zero.
               time_exponent_alpha: float = 0.5, # Most commonly used exponent in image generation is 1.0, but it is too slow in our case.
               ) -> Tuple[torch.Tensor, PointCloud, PointCloud]:
        """
        alpha = timestep * L^2 * (t^time_exponent_alpha)
        T = temperature * (t^time_exponent_temp)
        """

        if diffusion_schedules_list is None:
            diffusion_schedules_list = [None for _ in range(len(self.models))]
        assert len(self.models) == len(N_steps_list), f"{len(self.models)} != {len(N_steps_list)}"
        assert len(self.models) == len(timesteps_list), f"{len(self.models)} != {len(timesteps_list)}"
        assert len(self.models) == len(temperatures_list), f"{len(self.models)} != {len(temperatures_list)}"
        assert len(self.models) == len(diffusion_schedules_list), f"{len(self.models)} != {len(diffusion_schedules_list)}"

        scene_pcd: PointCloud = self.proc_fn(scene_pcd)
        grasp_pcd: PointCloud = self.proc_fn(grasp_pcd)
        Ts_init: SE3 = self.proc_fn(Ts_init)

        scene_input: FeaturedPoints = pcd_to_featured_points(scene_pcd)
        grasp_input: FeaturedPoints = pcd_to_featured_points(grasp_pcd)
        T0: torch.Tensor = Ts_init.poses
        assert T0.ndim == 2 and T0.shape[-1] == 7, f"{T0.shape}"

        Ts_out = []
        for model, N_steps, timesteps, temperatures, diffusion_schedules in zip(self.models, N_steps_list, timesteps_list, temperatures_list, diffusion_schedules_list):
            #################### Feature extraction #####################
            with torch.no_grad():
                scene_out_multiscale: List[FeaturedPoints] = model.get_key_pcd_multiscale(scene_input)
                grasp_out: FeaturedPoints = model.get_query_pcd(grasp_input)

            if diffusion_schedules is None:
                diffusion_schedules = model.diffusion_schedules
            assert len(diffusion_schedules) == len(N_steps), f"{len(diffusion_schedules)} != {len(N_steps)}"
            assert len(diffusion_schedules) == len(timesteps), f"{len(diffusion_schedules)} != {len(timesteps)}"

            #################### Sample #####################
            with torch.no_grad():
                Ts = model.sample(
                    T_seed=T0.clone().detach(),
                    scene_pcd_multiscale=scene_out_multiscale,
                    grasp_pcd=grasp_out,
                    diffusion_schedules=diffusion_schedules,
                    N_steps=N_steps,
                    timesteps=timesteps,
                    temperatures=temperatures,
                    log_t_schedule=log_t_schedule,
                    time_exponent_temp=time_exponent_temp,
                    time_exponent_alpha=time_exponent_alpha,
                )
                Ts = Ts.type(T0.dtype)
                T0 = Ts[-1]
                Ts_out.append(Ts)
        Ts_out = torch.cat(Ts_out, dim=0) # Ts_out: (nTime, nSample, 7)

        return Ts_out, scene_pcd, grasp_pcd



