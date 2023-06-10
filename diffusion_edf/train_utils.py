import os
from functools import partial
import shutil
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

from typing import List, Tuple, Optional, Union, Iterable, NamedTuple, Any, Sequence, Dict, Callable
from tqdm import tqdm
from beartype import beartype
import warnings

import torch
from torch.utils.data import DataLoader
from open3d.visualization.tensorboard_plugin import summary
from torch.utils.tensorboard import SummaryWriter
from torch_cluster import radius
from torch_scatter import scatter_sum

from edf_interface.data import PointCloud, SE3, TargetPoseDemo, DemoSequence
from edf_interface.data import preprocess
from diffusion_edf.gnn_data import FeaturedPoints, merge_featured_points, pcd_to_featured_points
from diffusion_edf.dist import diffuse_isotropic_se3_batched

def compose_proc_fn(preprocess_config: Dict) -> Callable:
    proc_fn = []
    for proc in preprocess_config:
        proc_fn.append(
            partial(getattr(preprocess, proc['name']), **proc['kwargs'])
        )
    proc_fn = preprocess.compose_procs(proc_fn)
    return proc_fn

def flatten_batch(demo_batch: List[TargetPoseDemo]) -> Tuple[FeaturedPoints, FeaturedPoints, torch.Tensor]:
    scene_pcd = []
    grasp_pcd = []
    target_poses = []
    for b, demo in enumerate(demo_batch):
        scene_pcd.append(pcd_to_featured_points(demo.scene_pcd,batch_idx=b))
        grasp_pcd.append(pcd_to_featured_points(demo.grasp_pcd,batch_idx=b))
        target_poses.append(demo.target_poses.poses)

    scene_pcd = merge_featured_points(scene_pcd) # Shape: x: (b*p, 3), f: (b*p, 3), b: (b*p, )   # b: N_batch, p: N_points_scene
    grasp_pcd = merge_featured_points(grasp_pcd) # Shape: x: (b*p, 3), f: (b*p, 3), b: (b*p, )   # b: N_batch, p: N_points_grasp
    target_poses = torch.stack(target_poses, dim=0) # Shape: (b, g, 4+3)                         # g: N_poses

    return scene_pcd, grasp_pcd, target_poses

def get_collate_fn(task, proc_fn):
    if task == 'pick':
        def collate_fn(data_batch: List[DemoSequence]) -> List[TargetPoseDemo]:
            return [proc_fn(demo_seq[0]) for demo_seq in data_batch]
    elif task == 'place':
        def collate_fn(data_batch: List[DemoSequence]) -> List[TargetPoseDemo]:
            return [proc_fn(demo_seq[1]) for demo_seq in data_batch]
    else:
        raise ValueError(f"Unknown task name: {task}")

    return collate_fn

def sample_reference_points(src_points: torch.Tensor, dst_points: torch.Tensor, r: float, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_dst, edge_src = radius(x=src_points, y=dst_points, r=r)
    n_points = len(dst_points)
    n_neighbor = scatter_sum(src=torch.ones_like(edge_dst), index=edge_dst, dim_size=n_points)
    total_count = n_neighbor.sum()
    if total_count <= 0:
        raise ValueError("There is no connected edges. Increase the clustering radius.")
    p_choice = n_neighbor / total_count

    sampled_idx = torch.multinomial(p_choice, num_samples=n_samples)
    return dst_points.index_select(0, sampled_idx), n_neighbor

@beartype
def transform_and_sample_reference_points(T_target: torch.Tensor,
                                          scene_points: FeaturedPoints,
                                          grasp_points: FeaturedPoints,
                                          contact_radius: Union[float, int],
                                          n_samples_x_ref: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert T_target.ndim == 2 and T_target.shape[-1] == 7, f"{T_target.shape}" # (nT, 7)
    if len(T_target) != 1:
        raise NotImplementedError
    
    x_ref, n_neighbors = sample_reference_points(
        src_points = PointCloud(points=scene_points.x, colors=scene_points.f).transformed(
                                SE3(T_target).inv(), squeeze=True
                                ).points, 
        dst_points = grasp_points.x, 
        r=float(contact_radius), 
        n_samples=n_samples_x_ref
    )
    return x_ref, n_neighbors

@beartype
def random_time(min_time: Union[float, int], 
                max_time: Union[float, int],
                device: Union[str, torch.device]) -> torch.Tensor:
    device = torch.device(device)
    assert min_time < max_time and min_time > 0.00001
    min_time = torch.tensor([float(min_time)], device=device)
    max_time = torch.tensor([float(max_time)], device=device)

    time = (min_time/max_time + torch.rand(1, device = min_time.device, dtype=min_time.dtype) * (1-min_time/max_time))*max_time   # Shape: (1,)
    #time = torch.exp(torch.rand_like(max_time) * (torch.log(max_time)-torch.log(min_time)) + torch.log(min_time)) 
    return time


@beartype
def diffuse_T_target(T_target: torch.Tensor, 
                     x_ref: torch.Tensor,
                     time: torch.Tensor,
                     lin_mult: Union[float, int],
                     ang_mult: Union[float, int] = 1.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    assert T_target.ndim == 2 and T_target.shape[-1] == 7, f"{T_target.shape}" # (nT, 7)
    if len(T_target) != 1:
        raise NotImplementedError
    assert x_ref.ndim == 2 and x_ref.shape[-1] == 3, f"{x_ref.shape}" # (n_xref, 7)
    if not time.shape == (1,):
        raise NotImplementedError

    eps = time / 2 * (float(ang_mult) ** 2)   # Shape: (1,)
    std = torch.sqrt(time) * float(lin_mult)   # Shape: (1,)

    # T, delta_T, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref) = diffuse_isotropic_se3(T0 = T_target, eps=eps, std=std, x_ref=x_ref, double_precision=True)
    T, delta_T, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref) = diffuse_isotropic_se3_batched(T0 = T_target, eps=eps, std=std, x_ref=x_ref, double_precision=True)
    T, delta_T, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref) = T.squeeze(-2), delta_T.squeeze(-2), (gt_ang_score.squeeze(-2), gt_lin_score.squeeze(-2)), (gt_ang_score_ref.squeeze(-2), gt_lin_score_ref.squeeze(-2))
    # T: (nT, 7) || delta_T: (nT, 7) || gt_*_score_*: (nT, 3) ||
    # Note that nT = n_samples_x_ref * nT_target  ||   nT_target = 1

    time_in = time.repeat(len(T))

    return T, delta_T, time_in, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref)


class LazyLogger():
    is_writer_online: bool
    log_dir: str
    configs_root_dir: Optional[str]

    def __init__(self, log_dir: str, resume: bool = False, configs_root_dir: Optional[str] = None):
        self.writer = None
        self.log_dir = log_dir
        self.resume = resume
        if not configs_root_dir:
            assert resume is True, f"Please provide dir to config files if you are not resuming from previous training."

        self.configs_root_dir = configs_root_dir

    @property 
    def is_initialized(self) -> bool:
        if self.writer is None:
            return False
        else:
            return True

    def _copy_configs(self, src_configs_root_dir: str, dst_configs_root_dir: str):
        assert not os.path.exists(dst_configs_root_dir), f'Config path "{dst_configs_root_dir}" already exists.'
        #os.mkdir(dst_configs_root_dir)
        for root, dirs, files in os.walk(src_configs_root_dir):
            for file in files:
                if file[-5:] != '.yaml':
                    raise FileExistsError(f"Non-config file \"{os.path.join(root, file)}\" found. config files must end with \".yaml\".")
        shutil.copytree(src_configs_root_dir, dst_configs_root_dir)

    def _lazy_init(self, log_dir: str, resume: bool, configs_root_dir: Optional[str]) -> bool:
        if self.is_initialized:
            raise ValueError("Already initialized")
        else:
            self.writer = SummaryWriter(log_dir=log_dir)

            checkpoint_dir = os.path.join(log_dir, f'checkpoint')
            if not os.path.exists(checkpoint_dir):
                if resume:
                    raise ValueError(f"Resuming from training, but cannot find the checkpoint dir: {checkpoint_dir}")
                else:
                    os.mkdir(checkpoint_dir)

            logged_configs_dir = os.path.join(log_dir, f'configs')
            if resume:
                if not os.path.exists(logged_configs_dir):
                    warnings.warn(f"Resuming from training, but cannot find the configs dir: {logged_configs_dir}")
            else:
                self._copy_configs(src_configs_root_dir=configs_root_dir, dst_configs_root_dir=logged_configs_dir)
                
            return True

    def lazy_init(self):
        if not self.is_initialized:
            self._lazy_init(log_dir=self.log_dir, resume=self.resume, configs_root_dir=self.configs_root_dir)
        
    def add_scalar(self, *args, **kwargs):
        self.lazy_init()
        self.writer.add_scalar(*args, **kwargs)

    def add_3d(self, *args, **kwargs):
        self.lazy_init()
        self.writer.add_3d(*args, **kwargs)

