import os
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

from typing import List, Tuple, Optional, Union, Iterable, NamedTuple, Any, Sequence, Dict
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

class LazyInitSummaryWriter():
    is_writer_online: bool
    log_dir: str

    def __init__(self, log_dir: str, resume: bool = False, config_files: Optional[List[str]] = None):
        self.writer = None
        self.log_dir = log_dir
        self.resume = resume
        if config_files is None:
            assert resume is True, f"Please provide dir to config_files if you are not resuming from previous training."

        self.config_files: Dict[str, str] = {}
        for file_path in config_files:
            with open(file_path) as file:
                filename = os.path.split(file_path)[-1]
                lines = file.read()
                self.config_files[filename] = lines


    def _lazy_init(self, 
                  log_dir: Optional[str] = None, 
                  resume: Optional[bool] = None,
                  config_files: Optional[Dict[str, str]] = None) -> bool:
        if self.writer is not None:
            return True
        else:
            if log_dir is None:
                log_dir = self.log_dir
            if resume is None:
                resume = self.resume
            if config_files is None:
                config_files = self.config_files
            self.writer = SummaryWriter(log_dir=log_dir)

            if not os.path.exists(os.path.join(log_dir, f'checkpoint')):
                if resume:
                    warnings.warn(f"Resuming from training, but cannot find the checkpoint dir: {os.path.join(log_dir, f'checkpoint')}")
                os.mkdir(os.path.join(log_dir, f'checkpoint'))
            if not os.path.exists(os.path.join(log_dir, f'configs')):
                if resume:
                    warnings.warn(f"Resuming from training, but cannot find the configs dir: {os.path.join(log_dir, f'configs')}")
                os.mkdir(os.path.join(log_dir, f'configs'))

            if not resume:
                for filename, contents in config_files.items():
                    with open(os.path.join(log_dir, f'configs', filename), mode='w') as file:
                        file.write(contents)
            return True
        
    def add_scalar(self, *args, **kwargs):
        self._lazy_init()
        self.writer.add_scalar(*args, **kwargs)

    def add_3d(self, *args, **kwargs):
        self._lazy_init()
        self.writer.add_3d(*args, **kwargs)

