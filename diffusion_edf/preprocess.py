from typing import Union, Optional, List, Tuple, Dict

import numpy as np

import torch

from diffusion_edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence
from diffusion_edf.pc_utils import voxel_filter, normalize_pc_color


def rescale(data, rescale_factor: float):
    if data is None:
        return None
    elif type(data) == DemoSequence:
        demo_seq = [rescale(demo, rescale_factor=rescale_factor) for demo in data]
        return DemoSequence(demo_seq = demo_seq, device=data.device)
    elif type(data) == TargetPoseDemo:
        scene_pc = rescale(data.scene_pc, rescale_factor=rescale_factor)
        grasp_pc = rescale(data.grasp_pc, rescale_factor=rescale_factor)
        target_poses = rescale(data.target_poses, rescale_factor=rescale_factor)
        return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=scene_pc.device, name=data.name)
    elif type(data) == PointCloud:
        if data.is_empty():
            return data
        else:
            points = data.points * rescale_factor
            return PointCloud(points=points, colors=data.colors.clone(), device=data.device)
    elif type(data) == SE3:
        poses = data.poses * torch.tensor([1., 1., 1., 1., rescale_factor, rescale_factor, rescale_factor], dtype=data.poses.dtype, device=data.poses.device).unsqueeze(-2)
        return SE3(poses=poses, device=data.device)
    else:
        raise TypeError(f"Unknown data type ({type(data)}) is given.")



def normalize_color(data, color_mean: torch.Tensor, color_std: torch.Tensor):
    if data is None:
        return None
    elif type(data) == DemoSequence:
        demo_seq = [normalize_color(demo, color_mean=color_mean, color_std=color_std) for demo in data]
        return DemoSequence(demo_seq = demo_seq, device=data.device)
    elif type(data) == TargetPoseDemo:
        scene_pc = normalize_color(data.scene_pc, color_mean=color_mean, color_std=color_std)
        grasp_pc = normalize_color(data.grasp_pc, color_mean=color_mean, color_std=color_std)
        target_poses = normalize_color(data.target_poses, color_mean=color_mean, color_std=color_std)
        return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
    elif type(data) == PointCloud:
        if data.is_empty():
            return data
        else:
            return normalize_pc_color(data=data, color_mean=color_mean, color_std=color_std)
    elif type(data) == SE3:
        return data
    else:
        raise TypeError(f"Unknown data type ({type(data)}) is given.")



def downsample(data, voxel_size: float, coord_reduction: str = "average"):
    if data is None:
        return None
    elif type(data) == DemoSequence:
        demo_seq = [downsample(demo, voxel_size=voxel_size, coord_reduction=coord_reduction) for demo in data]
        return DemoSequence(demo_seq = demo_seq, device=data.device)
    elif type(data) == TargetPoseDemo:
        scene_pc = downsample(data.scene_pc, voxel_size=voxel_size, coord_reduction=coord_reduction)
        grasp_pc = downsample(data.grasp_pc, voxel_size=voxel_size, coord_reduction=coord_reduction)
        target_poses = downsample(data.target_poses, voxel_size=voxel_size, coord_reduction=coord_reduction)
        return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
    elif type(data) == PointCloud:
        if data.is_empty():
            return data
        else:
            return voxel_filter(pc=data, voxel_size=voxel_size, coord_reduction=coord_reduction)
    elif type(data) == SE3:
        return data
    else:
        raise TypeError(f"Unknown data type ({type(data)}) is given.")



def apply_SE3(data, poses: SE3, apply_right: bool = False):
    if data is None:
        return None
    elif type(data) == DemoSequence:
        demo_seq = [apply_SE3(data=demo, poses=poses, apply_right=apply_right) for demo in data]
        return DemoSequence(demo_seq = demo_seq, device=data.device)
    elif type(data) == TargetPoseDemo:
        scene_pc = apply_SE3(data=scene_pc, poses=poses, apply_right=apply_right)
        grasp_pc = apply_SE3(data=grasp_pc, poses=poses, apply_right=apply_right)
        target_poses = apply_SE3(data=target_poses, poses=poses, apply_right=apply_right)
        return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
    elif type(data) == PointCloud:
        if data.is_empty():
            return data
        else:
            return data.transformed(poses)
    elif type(data) == SE3:
        return SE3.multiply(poses, data, apply_right=apply_right)
    else:
        raise TypeError(f"Unknown data type ({type(data)}) is given.")
    


def jitter_points(data, jitter_std: float):
    if data is None:
        return None
    elif type(data) == DemoSequence:
        demo_seq = [jitter_points(data=demo, jitter_std=jitter_std) for demo in data]
        return DemoSequence(demo_seq = demo_seq, device=data.device)
    elif type(data) == TargetPoseDemo:
        scene_pc = jitter_points(data=scene_pc, jitter_std=jitter_std)
        grasp_pc = jitter_points(data=grasp_pc, jitter_std=jitter_std)
        target_poses = jitter_points(data=target_poses, jitter_std=jitter_std)
        return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
    elif type(data) == PointCloud:
        if data.is_empty():
            return data
        else:
            points = data.points + torch.randn(*(data.points.shape), device=data.points.device, dtype=data.points.dtype) * jitter_std
            return PointCloud(points=points, colors=data.colors)
    elif type(data) == SE3:
        return data
    else:
        raise TypeError(f"Unknown data type ({type(data)}) is given.")



def jitter_colors(data, jitter_std: float, cutoff: bool = False):
    if data is None:
        return None
    elif type(data) == DemoSequence:
        demo_seq = [jitter_points(data=demo, jitter_std=jitter_std, cutoff=cutoff) for demo in data]
        return DemoSequence(demo_seq = demo_seq, device=data.device)
    elif type(data) == TargetPoseDemo:
        scene_pc = jitter_points(data=scene_pc, jitter_std=jitter_std, cutoff=cutoff)
        grasp_pc = jitter_points(data=grasp_pc, jitter_std=jitter_std, cutoff=cutoff)
        target_poses = jitter_points(data=target_poses, jitter_std=jitter_std, cutoff=cutoff)
        return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
    elif type(data) == PointCloud:
        if data.is_empty():
            return data
        else:
            colors = data.colors + torch.randn(*(data.colors.shape), device=data.colors.device, dtype=data.colors.dtype) * jitter_std
            if cutoff:
                colors = torch.where(colors > 1., torch.tensor(1., dtype=colors.dtype, device=colors.device), colors)
                colors = torch.where(colors < 0., torch.tensor(0., dtype=colors.dtype, device=colors.device), colors)
            return PointCloud(points=data.points, colors=colors)
    elif type(data) == SE3:
        return data
    else:
        raise TypeError(f"Unknown data type ({type(data)}) is given.")



class EdfTransform(torch.nn.Module):
    def __init__(self, supported_type):
        super().__init__()
        self.supported_type = supported_type

    def forward(self, data):
        if type(data) not in self.supported_type:
            raise TypeError
        return data



class Rescale(EdfTransform):
    def __init__(self, rescale_factor: float) -> None:
        super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
        self.rescale_factor: float = rescale_factor

    def forward(self, data):
        data = super().forward(data)
        return rescale(data=data, rescale_factor = self.rescale_factor)
        

    
class NormalizeColor(EdfTransform):
    def __init__(self, color_mean: torch.Tensor, color_std: torch.Tensor) -> None:
        super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
        self.color_mean: torch.Tensor = color_mean
        self.color_std: torch.Tensor = color_std

    def forward(self, data):
        data = super().forward(data)
        return normalize_color(data=data, color_mean = self.color_mean, color_std = self.color_std)


    
class Downsample(EdfTransform):
    def __init__(self, voxel_size: float, coord_reduction: str = "average") -> None:
        super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
        self.voxel_size: float = voxel_size
        self.coord_reduction: str = coord_reduction

    def forward(self, data):
        data = super().forward(data)
        return downsample(data=data, voxel_size=self.voxel_size, coord_reduction=self.coord_reduction)
    
    def extra_repr(self) -> str:
        return f"voxel_size: {self.voxel_size}, coord_reduction: {self.coord_reduction}"


    
class ApplySE3(EdfTransform):
    def __init__(self, poses: SE3, apply_right: bool = False) -> None:
        super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
        self.poses = SE3(poses=poses.detach().clone())
        self.apply_right = apply_right

    def forward(self, data):
        data = super().forward(data)
        return apply_SE3(data=data, poses=self.poses, apply_right=self.apply_right)



class PointJitter(EdfTransform):
    def __init__(self, jitter_std: float) -> None:
        super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
        self.jitter_std: float = jitter_std

    def forward(self, data):
        data = super().forward(data)
        return jitter_points(data=data, jitter_std=self.jitter_std)
    


class ColorJitter(EdfTransform):
    def __init__(self, jitter_std: float, cutoff: bool = False) -> None:
        super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
        self.jitter_std: float = jitter_std
        self.cutoff: bool = cutoff

    def forward(self, data):
        data = super().forward(data)
        return jitter_colors(data=data, jitter_std=self.jitter_std, cutoff=self.cutoff)