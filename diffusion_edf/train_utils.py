import os
import shutil
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

from typing import List, Tuple, Optional, Union, Iterable, NamedTuple, Any, Sequence, Dict, Callable
from tqdm import tqdm
from beartype import beartype
import warnings

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from open3d.visualization.tensorboard_plugin import summary
from torch.utils.tensorboard import SummaryWriter

from diffusion_edf.data import DemoSeqDataset, DemoSequence, TargetPoseDemo
from diffusion_edf.gnn_data import FeaturedPoints, merge_featured_points, pcd_to_featured_points
from diffusion_edf import preprocess

def compose_proc_fn(preprocess_config: Dict) -> Callable:
    proc_fn = []
    for proc in preprocess_config:
        proc_fn.append(
            getattr(preprocess, proc['name'])(**proc['kwargs'])
        )
    proc_fn = Compose(proc_fn)
    return proc_fn

def flatten_batch(demo_batch: List[TargetPoseDemo]) -> Tuple[FeaturedPoints, FeaturedPoints, torch.Tensor]:
    scene_pcd = []
    grasp_pcd = []
    target_poses = []
    for b, demo in enumerate(demo_batch):
        scene_pcd.append(pcd_to_featured_points(demo.scene_pc,batch_idx=b))
        grasp_pcd.append(pcd_to_featured_points(demo.grasp_pc,batch_idx=b))
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

