from typing import Optional, Tuple, Union, Iterable

import open3d as o3d
import numpy as np
import plotly.graph_objects as go

import torch
import torch.nn.functional as F
import torch_scatter
import torch_cluster
from pytorch3d.transforms import quaternion_apply, quaternion_multiply, axis_angle_to_quaternion

from diffusion_edf.data import PointCloud, SE3



def get_plotly_fig(title: Optional[str] = None, width: int=600, height: int=600) -> go.Figure:
    fig: go.Figure = go.Figure()
    fig.update_layout(showlegend=False)
    fig.update_layout(scene = dict(xaxis = dict(visible=False),
                                yaxis = dict(visible=False),
                                zaxis = dict(visible=False),))
    fig.update_layout(scene_aspectmode='data', width=width, height=height, 
                      margin=dict(l=10, r=10, b=10, t=40, pad=1))
    fig.update_layout(title=title)
    return fig


def pcd_from_numpy(coord: np.ndarray, color: Optional[np.ndarray], voxel_filter_size: Optional[float] = None):
    assert len(coord.shape) == 2, f"coord must be of shape (N_points, 3), but shape {coord.shape} is given."
    if color is None:
        raise NotImplementedError
        color = np.tile(np.array([[0.8, 0.5, 0.8]]), (coord.shape[-2],1)) # (N_points, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)

    if voxel_filter_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_filter_size)

    return pcd

def pcd_to_numpy(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    return points, colors

def transform_points(points: torch.Tensor, Ts: torch.Tensor) -> torch.Tensor:
    ndim = Ts.ndim
    assert ndim <= 2 and Ts.shape[-1] == 7
    assert points.ndim == 2 and points.shape[-1] == 3, f"Points must have shape (N,3), but shape {points.shape} is given."
    assert points.device == Ts.device
    
    q, t = Ts[..., :4], Ts[..., 4:]
    N_points: int = points.shape[-2]
    if Ts.ndim == 2:
        N_transforms: int = len(Ts)
        points = quaternion_apply(quaternion=q.unsqueeze(-2).expand([-1,N_points,-1]), point=points.unsqueeze(-3).expand([N_transforms,-1,-1])) + t.unsqueeze(-2) # (N_transforms, N_points, 3)
    else:
        points = quaternion_apply(quaternion=q.unsqueeze(-2).expand([N_points,-1]), point=points) + t # (N_points, 3)
    return points


def draw_geometry(geometries):
    if not hasattr(geometries, '__iter__'):
        geometries = [geometries]
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in geometries:
        if type(geometry) == PointCloud:
            geometry = geometry.to_pcd()
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.8, 0.8, 0.8])
    viewer.run()
    viewer.destroy_window()

def create_o3d_points(points: torch.Tensor, colors: Union[str, Iterable] = 'blue', radius = 0.5, alpha: Optional[Iterable] = None):
    if type(colors) == str:
        if colors=='blue':
            colors=[0.1, 0.7, 0.7]
        elif colors=='red':
            colors=[0.7, 0.1, 0.1]
        else:
            raise ValueError("Unknown color name")
    else:
        assert isinstance(colors, Iterable)
        assert len(colors) == 3        

    points_visual = []
    for point in points.detach().clone().cpu():
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(colors)
        mesh_sphere.translate(point)
        points_visual.append(mesh_sphere)
    return points_visual

def voxel_filter(pc: PointCloud, voxel_size: float, coord_reduction: str = "average") -> PointCloud:
    device = pc.device

    mins = pc.points.min(dim=-2).values

    vox_idx = torch.div((pc.points - mins), voxel_size, rounding_mode='trunc').type(torch.long)
    shape = vox_idx.max(dim=-2).values + 1
    raveled_idx = torch.tensor(np.ravel_multi_index(vox_idx.T.cpu().numpy(), shape.cpu().numpy()), device = device, dtype=vox_idx.dtype)

    n_pts_per_vox = torch_scatter.scatter(torch.ones_like(raveled_idx, device=device), raveled_idx, dim_size=shape[0]*shape[1]*shape[2])
    nonzero_vox = n_pts_per_vox.nonzero()
    n_pts_per_vox = n_pts_per_vox[nonzero_vox].squeeze(-1)

    color_vox = torch_scatter.scatter(pc.colors, raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
    color_vox /= n_pts_per_vox.unsqueeze(-1)

    
    if coord_reduction == "center":
        coord_vox = np.stack(np.unravel_index(nonzero_vox.cpu().numpy().reshape(-1), shape.cpu().numpy()), axis=-1)
        coord_vox = torch.tensor(coord_vox, device = device, dtype=vox_idx.dtype)
        coord_vox = coord_vox * voxel_size + mins + (voxel_size/2)
    elif coord_reduction == "average":
        coord_vox = torch_scatter.scatter(pc.points, raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
        coord_vox /= n_pts_per_vox.unsqueeze(-1)
    else:
        raise ValueError(f"Unknown coordinate reduction method: {coord_reduction}")

    return PointCloud(points=coord_vox, colors=color_vox, device=pc.device)

def normalize_pc_color(data: PointCloud, color_mean: torch.Tensor, color_std: torch.Tensor):
    assert type(data) == PointCloud, f"data must be of PointCloud type, but {type(data)} is given."
    colors = (data.colors - color_mean.to(data.device)) / color_std.to(data.device)

    return PointCloud(points=data.points.clone(), colors = colors, device=data.device)


# def reconstruct_surface(data: PointCloud) -> o3d.geometry.TriangleMesh:
#     pcd = data.to_pcd()
#     pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
#                                             std_ratio=2.0)
    
#     alpha = 0.015
#     mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
#     # mesh.compute_vertex_normals()
#     # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
#     return mesh

def check_pcd_collision(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], r: float) -> bool:
    if isinstance(x, PointCloud):
        x = x.points
    if isinstance(y, PointCloud):
        y = y.points

    return torch_cluster.radius(x=x, y=y, r = r).any().item()


def pcd_energy(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, grad: bool =True) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    if isinstance(x, PointCloud):
        x = x.points
    if isinstance(y, PointCloud):
        y = y.points

    radius_graph = torch_cluster.radius(x=x, y=y, r = cutoff_r)
    n_edge = radius_graph.shape[-1]

    if n_edge == 0:
        if grad is True:
            return torch.tensor(0., device=x.device), torch.zeros(3, device=x.device), n_edge
        else:
            return torch.tensor(0., device=x.device), None, n_edge

    x = x[radius_graph[1]].detach()
    y = y[radius_graph[0]].detach()

    if grad is True:
        dist_y = torch.zeros(3, requires_grad=True, device=x.device)
        y = y + dist_y

    energy = (cutoff_r/(x-y).norm(dim=-1, p=1)).sum(dim=-1)

    if grad is True:
        energy.backward()
        return energy.detach(), dist_y.grad.detach(), n_edge
    else:
        return energy.detach(), None, n_edge
    
def _optimize_pcd_collision_once(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, dt: float, eps: float) -> Tuple[Union[PointCloud, torch.Tensor], SE3, bool]:
    if isinstance(x, PointCloud):
        x = x.points
    if isinstance(y, PointCloud):
        pcd = y
        y = y.points
    else:
        pcd = None

    energy, grad, n_edge = pcd_energy(x=x, y=y, cutoff_r=cutoff_r, grad=True)

    if n_edge == 0:
        done = True
    else:
        done = False

    disp = -grad / (grad.norm() + eps) * dt
    disp_pose = F.pad(F.pad(disp, pad=(3,0), value=0.), pad=(1,0), value=1.) # (1,0,0,0,x,y,z)
    disp_pose = SE3(disp_pose)

    if pcd is None:
        return y + disp, disp_pose, done
    else:
        return pcd.transformed(disp_pose)[0], disp_pose, done
    
def optimize_pcd_collision(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, dt: float, eps: float, iters: int, rel_pose: Optional[SE3] = None) -> Tuple[Union[PointCloud, torch.Tensor], SE3]:
    if rel_pose is not None:
        if isinstance(y, PointCloud):
            y = y.transformed(Ts=rel_pose)[0]
        else:
            raise NotImplementedError

    Ts = []
    for _ in range(iters):
        y, T, done = _optimize_pcd_collision_once(x=x, y=y, cutoff_r=cutoff_r, dt=dt, eps=eps)
        Ts.append(T)
        if done:
            break

    if rel_pose is not None:
        Ts.append(rel_pose)
    return y, SE3.multiply(*Ts)
