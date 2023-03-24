from __future__ import annotations
import os
import gzip
import pickle
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
import open3d as o3d
import yaml
import plotly.graph_objects as go

import torch
from torchvision.transforms import Compose
from pytorch3d.transforms import quaternion_apply, quaternion_multiply, axis_angle_to_quaternion, quaternion_invert



def load_yaml(file_path: str):
    """Loads yaml file from path."""
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def gzip_save(data, path: str):
    dir = os.path.dirname(path)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)

def gzip_load(path: str):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class SE3():
    def __init__(self, poses: Union[torch.Tensor, Iterable], device: Optional[Union[str, torch.device]] = None, renormalize: bool = True):
        if not isinstance(poses, torch.Tensor):
            if device is None:
                device = torch.device('cpu')
            poses = torch.tensor(poses, device=device)

        assert poses.ndim <= 2 and poses.shape[-1] == 7

        if device is None:
            device = poses.device
        device = torch.device(device)
        if device != poses.device:
            poses = poses.to(device)
        
        if poses.ndim == 1:
            poses = poses.unsqueeze(-2)

        self.device = device
        self.poses = poses

        if not torch.allclose(self.poses[...,:4].detach().norm(dim=-1,keepdim=True), torch.tensor([1.0], device=device), rtol=0, atol=0.03):
            warnings.warn("SE3.__init__(): Input quaternion is not normalized")

        if renormalize:
            self.poses[...,:4] = self.poses[...,:4] / self.poses[...,:4].norm(dim=-1,keepdim=True)

        self.inv = self._inv

    def to(self, device: Union[str, torch.device]) -> SE3:
        device = torch.device(device)
        if device == self.device:
            return self
        else:
            return SE3(poses=self.poses.to(device), device=device)

    @staticmethod
    def from_orn_and_pos(orns: torch.Tensor, positions: torch.Tensor, versor_last_input: bool = False, device: Optional[Union[str, torch.device]] = None) -> SE3:
        assert positions.ndim == 2 and orns.ndim == 2 and positions.shape[-1] == 3 and orns.shape[-1] == 4
        assert positions.shape[-2] == orns.shape[-2]

        if device is None:
            assert orns.device == positions.device
            device = positions.device
        else:
            device = torch.device(device)
            orns = orns.to(device)
            positions = positions.to(device)

        if versor_last_input:
            poses = torch.cat((orns[..., 3:4], orns[..., :3], positions), dim=-1)
        else:
            poses = torch.cat((orns, positions), dim=-1)

        return SE3(poses=poses, device=device)

    @staticmethod
    def from_numpy(orns: np.ndarray, positions: np.ndarray, versor_last_input: bool = False, device: Union[str, torch.device] = 'cpu') -> SE3:
        return SE3.from_orn_and_pos(orns = torch.tensor(orns, dtype=torch.float32, device=device), 
                                    positions=torch.tensor(positions, dtype=torch.float32, device=device), 
                                    versor_last_input=versor_last_input, 
                                    device=device)

    def __len__(self) -> int:
        return len(self.poses)

    def __repr__(self) -> str:
        string = self.poses.__repr__().lstrip("tensor")
        return f"SE(3) with {self.__len__()} poses\n    - {string}"

    def __str__(self) -> str:
        string = self.poses.__str__().lstrip("tensor")
        return f"SE(3) with {self.__len__()} poses\n    - {string}"

    def get_data_dict(self) -> Dict[str, torch.Tensor]:
        data_dict = {"poses": self.poses.detach().cpu()}
        return data_dict

    @staticmethod
    def from_data_dict(data_dict: Dict, device: Union[str, torch.device] = 'cpu') -> SE3:
        assert "poses" in data_dict.keys() 
        return SE3(poses=data_dict["poses"].to(device), device=device)
    
    @staticmethod
    def multiply(*Ts) -> SE3:
        T: SE3 = Ts[-1]
        q,x = T.poses[...,:4], T.poses[...,4:]
        q = q / q.norm(dim=-1, keepdim=True)

        for T in Ts[-2::-1]:
            assert len(T.poses) == len(q) == len(x) or len(T.poses) == 1 or len(q) == len(x) == 1
            x = quaternion_apply(T.poses[...,:4], x) + T.poses[...,4:]
            q = quaternion_multiply(T.poses[...,:4], q)
            q = q / q.norm(dim=-1, keepdim=True)
        return SE3(poses=torch.cat([q,x], dim=-1), renormalize=False)
    
    @staticmethod
    def inv(T) -> SE3:
        q, x = T.poses[...,:4], T.poses[...,4:]
        q_inv = quaternion_invert(q / q.norm(dim=-1, keepdim=True))
        q_inv = q_inv / q_inv.norm(dim=-1, keepdim=True)
        x_inv = -quaternion_apply(q_inv, x)
        return SE3(poses=torch.cat([q_inv, x_inv], dim=-1), renormalize=False)
    
    def _inv(self) -> SE3:
        return SE3.inv(self)
    
    def __mul__(self, other) -> SE3:
        return SE3.multiply(self, other)
    
    def __getitem__(self, idx) -> SE3:
        assert type(idx) == slice or type(idx) == int, "Indexing must be an integer or a slice with single axis."
        return SE3(poses=self.poses[idx], renormalize=False)
    
    @property
    def orns(self) -> torch.Tensor:
        return self.poses[...,:4]
    
    @property
    def points(self) -> torch.Tensor:
        return self.poses[...,4:]
    
    # @staticmethod
    # def apply(T: SE3, points: Union[torch.Tensor, PointCloud]):
    #     if type(points) == PointCloud:
    #         is_pcd = True
    #         pcd = points
    #         points = pcd.points
    #     else:
    #         is_pcd = False

    #     assert points.shape[-1] == 3, "points must be 3-dimensional."
    #     assert T.device == points.device, f"device of SE(3) ({T.device}) does not matches device of points ({points.device})."
    #     points = (quaternion_apply(T.orns, points.view(-1, 3)) + T.points).view(*(points.shape))
    #     if is_pcd:
    #         points = PointCloud(points=points, colors=pcd.colors)
    #     return points

    @staticmethod
    def empty(device: Union[str, torch.device] = 'cpu') -> SE3:
        return SE3(poses=torch.empty((0,7), device=device), device=device)
    
    def is_empty(self) -> bool:
        if len(self.poses) == 0:
            return True
        else:
            return False


class PointCloud():
    def __init__(self, points: torch.Tensor, colors: torch.Tensor, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            device = points.device
        device = torch.device(device)
        assert points.device == colors.device
        self.device = device

        self.points: torch.Tensor = points # (N,3)
        self.colors: torch.Tensor = colors # (N,3)

    def to(self, device: Union[str, torch.device]) -> PointCloud:
        device = torch.device(device)
        if device == self.device:
            return self
        else:
            return PointCloud(points=self.points.to(device), colors=self.colors.to(device), device=device)

    def __repr__(self) -> str:
        return f"Colored point cloud with {len(self.points)} points."
    
    def __str__(self) -> str:
        return f"Colored point cloud with {len(self.points)} points."
    
    def __len__(self) -> int:
        return len(self.points)

    @staticmethod
    def from_numpy(points: np.ndarray, colors: np.ndarray, device: Union[str, torch.device] = 'cpu') -> PointCloud:
        return PointCloud(points=torch.tensor(points, dtype=torch.float32, device=device), colors=torch.tensor(colors, dtype=torch.float32, device=device), device=device)

    @staticmethod
    def from_o3d(pcd: o3d.geometry.PointCloud, device: Union[str, torch.device] = 'cpu') -> PointCloud:
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=device)

        return PointCloud(points=points, colors=colors, device=device)

    def to_o3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.detach().cpu())
        pcd.colors = o3d.utility.Vector3dVector(self.colors.detach().cpu())

        return pcd
    
    def get_data_dict(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        data_dict = {"points": self.points.detach().cpu(),
                     "colors": self.colors.detach().cpu()}
        return data_dict

    @staticmethod
    def from_data_dict(data_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]], device: Union[str, torch.device] = 'cpu') -> PointCloud:
        assert "points" in data_dict.keys() and "colors" in data_dict.keys()
        return PointCloud(points=data_dict["points"].to(device), colors=data_dict["colors"].to(device), device=device)
    
    @staticmethod
    def empty(device: Union[str, torch.device] = 'cpu') -> PointCloud:
        return PointCloud(points=torch.empty((0,3), device=device), colors=torch.empty((0,3), device=device))
    
    def is_empty(self) -> bool:
        if len(self.points) == 0:
            return True
        else:
            return False
    
    @staticmethod
    def transform_pcd(pcd, Ts: Union[torch.Tensor, SE3]) -> List[PointCloud]:
        assert isinstance(pcd, PointCloud)
        if isinstance(Ts, SE3):
            Ts = Ts.poses.detach().clone()
        ndim = Ts.ndim
        assert ndim <= 2 and Ts.shape[-1] == 7
        assert pcd.device == Ts.device

        from diffusion_edf.pc_utils import transform_points
        
        points = transform_points(points = pcd.points, Ts = Ts)
        if points.ndim == 3:
            output: List[PointCloud] = [PointCloud(points=point, colors=pcd.colors) for point in points]
        else:
            output: List[PointCloud] = [PointCloud(points=points, colors=pcd.colors)]
        return output

    def transformed(self, Ts: Union[torch.Tensor, SE3]) -> PointCloud:
        return PointCloud.transform_pcd(pcd=self, Ts=Ts)
    
    @staticmethod
    def merge(*args) -> PointCloud:
        points = []
        colors = []
        for pcd in args:
            points.append(pcd.points)
            colors.append(pcd.colors)
        points = torch.cat(points, dim=-2)
        colors = torch.cat(colors, dim=-2)

        return PointCloud(points=points, colors=colors)

    @staticmethod
    def points_to_plotly(pcd: Union[PointCloud, torch.Tensor], point_size: float = 1.0, name: Optional[str] = None, opacity: Union[float, torch.Tensor] = 1.0, colors: Optional[Iterable] = None, custom_data: Optional[Dict] = None) -> go.Scatter3d:
        if colors is not None:
            colors = torch.tensor(colors)
        if isinstance(pcd, PointCloud):
            points: torch.Tensor = pcd.points
            if colors is None:
                colors: torch.Tensor = pcd.colors
        if isinstance(pcd, torch.Tensor):
            assert pcd.ndim==2 and pcd.shape[-1] == 3, f"pcd must be 3-dimensional pointcloud, but pcd with shape {pcd.shape} is given."
            points: torch.Tensor = pcd
            if colors is None:
                colors: torch.Tensor = torch.zeros(len(points), 3)
        colors = colors.detach().cpu()
        if colors.ndim==1:
            colors = colors.expand(len(pcd), 3)

        pcd_marker = {}

        if isinstance(opacity, torch.Tensor):
            assert len(opacity) == pcd.__len__()
            colors = torch.cat([colors, opacity.detach().cpu().unsqueeze(-1)], dim=-1)
        elif type(opacity) == float:
            pcd_marker['opacity'] = opacity

        pcd_marker['size'] = point_size
        pcd_marker['color'] = colors


        plotly_kwargs = dict(x=points[:,0].detach().cpu(), y=points[:,1].detach().cpu(), z=points[:,2].detach().cpu(), mode='markers', marker=pcd_marker)
        if name is not None:
            plotly_kwargs['name'] = name

        if custom_data is not None:
            _customdata = []
            hover_template = ''
            for i,(k,v) in enumerate(custom_data.items()):
                _customdata.append(v)
                hover_template += f'<b>{k}</b>: ' + '%{customdata' +f'[{i}]' + ':,.2f}<br>'
            hover_template = hover_template.lstrip('<br>')
            hover_template += '<extra></extra>'
            plotly_kwargs['hovertemplate'] = hover_template
            print(hover_template)
            plotly_kwargs['customdata'] = torch.stack(_customdata, dim=-2)

        return go.Scatter3d(**plotly_kwargs)
    
    def plotly(self, point_size: float = 1.0, name: Optional[str] = None, opacity: Union[float, torch.Tensor] = 1.0, colors: Optional[torch.Tensor] = None, custom_data: Optional[dict] = None) -> go.Scatter3d:
        return PointCloud.points_to_plotly(pcd=self, point_size=point_size, name=name, opacity=opacity, colors=colors, custom_data=custom_data)

        

        
        





class Demo(metaclass=ABCMeta): 
    name: Optional[str] = None

    @abstractmethod
    def to(self, device: Union[str, torch.device]) -> Demo:
        pass
    
    @abstractmethod
    def get_data_dict(self) -> Dict[str, Any]:
        pass
    
    @staticmethod
    @abstractmethod
    def from_data_dict(data_dict: Dict[str, Any], device: Union[str, torch.device]):
        pass


class TargetPoseDemo(Demo):
    def __init__(self, target_poses: SE3,
                 scene_pc: Optional[PointCloud] = None, 
                 grasp_pc: Optional[PointCloud] = None,
                 name: Optional[str] = None,
                 device: Optional[Union[str, torch.device]] = None):
        
        self.name = name

        if device is None:
            device = target_poses.device
        device = torch.device(device)
        
        if scene_pc is None:
            scene_pc = PointCloud.empty(device=device)
        if grasp_pc is None:
            grasp_pc = PointCloud.empty(device=device)

        assert scene_pc.device == target_poses.device == grasp_pc.device == device
        self.device: torch.device = device


        self.scene_pc: PointCloud = scene_pc
        self.target_poses: SE3 = target_poses
        self.grasp_pc: PointCloud = grasp_pc

    def to(self, device: Union[str, torch.device]) -> TargetPoseDemo:
        device = torch.device(device)
        if device == self.device:
            return self
        else:
            return TargetPoseDemo(scene_pc=self.scene_pc.to(device=device), target_poses=self.target_poses.to(device=device), grasp_pc=self.grasp_pc.to(device=device), name=self.name, device=device)

    
    def get_data_dict(self) -> Dict:
        data_dict = {'name': self.name}
        for k,v in self.scene_pc.get_data_dict().items():
            data_dict["scene_"+k] = v
        for k,v in self.grasp_pc.get_data_dict().items():
            data_dict["grasp_"+k] = v
        for k,v in self.target_poses.get_data_dict().items():
            data_dict["target_"+k] = v
        data_dict["demo_type"]: str = self.__class__.__name__
        return data_dict

    @staticmethod
    def from_data_dict(data_dict: Dict, device: Union[str, torch.device] = 'cpu', rename: Union[bool, Optional[str]] = False):
        assert data_dict["demo_type"] == TargetPoseDemo.__name__
        scene_pc = PointCloud.from_data_dict(data_dict={"points": data_dict["scene_points"], "colors": data_dict["scene_colors"]}, device=device)
        grasp_pc = PointCloud.from_data_dict(data_dict={"points": data_dict["grasp_points"], "colors": data_dict["grasp_colors"]}, device=device)
        target_poses = SE3.from_data_dict(data_dict={"poses": data_dict["target_poses"]}, device=device)

        if rename is False:
            try:
                name = data_dict['name']
            except KeyError:
                name = ''
        elif rename is True:
            raise ValueError(f"The 'rename' argument must be a boolean 'False', or 'None', or string, but {rename} is given.")
        else:
            assert (isinstance(rename, str) or rename is None), f"The 'rename' argument must be a boolean 'False', or 'None', or string, but {rename} is given."
            name = rename

        return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, name = name, device=device)


class DemoSequence():
    valid_demo_type = ['TargetPoseDemo']
    str_to_demo_class = {'TargetPoseDemo': TargetPoseDemo}

    def __init__(self, demo_seq: List[Demo] = [], device: Optional[Union[str, torch.device]] = None):
        if device is None:
            if demo_seq:
                device = demo_seq[0].device
            else:
                device = "cpu"
        device = torch.device(device)
        self.device: torch.device = device

        self.demo_seq: List[Demo] = demo_seq


    def to(self, device: Union[str, torch.device]) -> DemoSequence:
        device = torch.device(device)
        if device == self.device:
            return self
        else:
            demo_seq: List[Demo] = []
            for demo in self.demo_seq:
                demo_seq.append(demo.to(device=device))
            return DemoSequence(demo_seq=demo_seq, device=device)
    
    def __getitem__(self, idx) -> Union[Demo, List[Demo]]:
        return self.demo_seq[idx]

    def get_data_seq(self) -> List[Dict]:
        return [demo.get_data() for demo in self.demo_seq]

    def save_data(self, path: str) -> bool:
        gzip_save(data=self.get_data_seq(), path = path)
        return True

    @staticmethod
    def from_data_seq(data_dict_seq: List[Dict], device: Union[str, torch.device] = 'cpu'):
        demo_seq: List[Demo] = []
        for data_dict in data_dict_seq:
            demo_type: str = data_dict["demo_type"]
            if demo_type in DemoSequence.valid_demo_type:
                demo_type: Type[Demo] = DemoSequence.str_to_demo_class[demo_type]
            else:
                raise ValueError(f"Unknown demo type: {demo_type}")
            demo_seq.append(demo_type.from_data_dict(data_dict=data_dict, device=device))
        
        return DemoSequence(demo_seq=demo_seq, device=device)

    @staticmethod    
    def from_file(path: str, device: Union[str, torch.device] = 'cpu') -> DemoSequence:
        data_dict_seq: List[Dict] = gzip_load(path=path)
        return DemoSequence.from_data_seq(data_dict_seq=data_dict_seq, device=device)
    

def save_demos(demos: List[DemoSequence], dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    with open(os.path.join(dir, "data.yaml"), 'w') as f:
        for i, demo in enumerate(demos):
            data_dir = "data"
            filename = f"demo_{i}.gzip"
            demo.save_data(path=os.path.join(dir, data_dir, filename))
            f.write("- \""+os.path.join(data_dir, filename)+"\"\n")

def load_demos(dir: str, annotation_file = "data.yaml") -> List[DemoSequence]:
    files = load_yaml(file_path=os.path.join(dir, annotation_file))

    demos: List[DemoSequence] = []
    for file in files:
        demos.append(DemoSequence.from_file(os.path.join(dir, file)))

    return demos


class DemoSeqDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str, 
                 annotation_file: str = "data.yaml", 
                 load_transforms: Optional[Union[Compose, torch.nn.Module]] = None, 
                 transforms: Optional[Union[Compose, torch.nn.Module]] = None, 
                 device: Union[str, torch.device] = 'cpu'):
        device = torch.device(device)
        if device != torch.device('cpu'):
            #raise NotImplementedError
            pass
        
        self.device = device
        self.load_transforms = load_transforms if load_transforms else lambda x:x
        self.transforms = transforms if transforms else lambda x:x

        self.data: List[DemoSequence] = [self.load_transforms(demo).to(self.device) for demo in load_demos(dir = dataset_dir, annotation_file=annotation_file)]

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     data = self.data[idx]
    #     return {'raw': data, 'processed': self.transforms(data)}

    def __getitem__(self, idx):
        data = self.transforms(self.data[idx])
        return data
    



    

        

    