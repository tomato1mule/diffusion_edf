# import os
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

from typing import List, Tuple, Optional, Union, Iterable, NamedTuple, Any, Sequence
from tqdm import tqdm

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

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