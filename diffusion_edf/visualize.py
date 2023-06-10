from typing import Tuple, Dict, List, Optional, Union

import torch
from edf_interface.data import PointCloud, SE3
from edf_interface.data.pcd_utils import get_plotly_fig

def visualize_pose(scene_pcd: PointCloud, 
                   grasp_pcd: PointCloud, 
                   poses: SE3, 
                   query: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                   show_sample_points: bool = False, 
                   point_size = 3.0, width=800, height=800,
                   ranges: Optional[torch.Tensor] = None):
    
    grasp_pl = grasp_pcd.plotly(point_size=point_size, name="grasp")
    grasp_geometry = [grasp_pl]
    if query is not None:
        query_points, query_attention = query
        query_opacity = query_attention ** 1
        query_pl = PointCloud.points_to_plotly(pcd=query_points, point_size=15.0, opacity=query_opacity / query_opacity.max())#, custom_data={'attention': query_attention.cpu()})
        grasp_geometry.append(query_pl)
    fig_grasp = get_plotly_fig("Grasp")
    fig_grasp = fig_grasp.add_traces(grasp_geometry)



    
    scene_pl = scene_pcd.plotly(point_size=point_size, name='scene')
    placement_geometry = [scene_pl]
    transformed_grasp_pcd = grasp_pcd.transformed(poses)

    if ranges is None:
        x_min = torch.min(scene_pcd.points[:,0]).item()
        x_max = torch.max(scene_pcd.points[:,0]).item()
        y_min = torch.min(scene_pcd.points[:,1]).item()
        y_max = torch.max(scene_pcd.points[:,1]).item()
        z_min = torch.min(scene_pcd.points[:,2]).item()
        z_max = torch.max(scene_pcd.points[:,2]).item()

    for i in range(len(poses)):
        if ranges is None:
            x_min = min(x_min, torch.min(transformed_grasp_pcd[i].points[:,0]).item())
            x_max = max(x_max, torch.max(transformed_grasp_pcd[i].points[:,0]).item())
            y_min = min(y_min, torch.min(transformed_grasp_pcd[i].points[:,1]).item())
            y_max = max(x_max, torch.max(transformed_grasp_pcd[i].points[:,1]).item())
            z_min = min(z_min, torch.min(transformed_grasp_pcd[i].points[:,2]).item())
            z_max = max(x_max, torch.max(transformed_grasp_pcd[i].points[:,2]).item())
        pose_pl = transformed_grasp_pcd[i].plotly(point_size=point_size, name=f'pose_{i}')
        placement_geometry.append(pose_pl)
    if show_sample_points:
        sample_pl = PointCloud.points_to_plotly(pcd=poses.points, point_size=point_size * 5.0, colors=[0.2, 0.5, 0.8], name=f'sample_points')
        placement_geometry.append(sample_pl)
    fig_sample = get_plotly_fig("Sampled Placement")
    fig_sample = fig_sample.add_traces(placement_geometry)
    
    trace_dict = {}
    visiblility_list = []
    for i, trace in enumerate(fig_sample.data):
        trace_dict[trace.name] = i
        if trace.name[:4] == 'pose':
            trace.visible = False
            visiblility_list.append(False)
        else:
            visiblility_list.append(trace.visible)

    # Define sliders
    steps = []
    for i in range(len(poses)):
        step = dict(
            method="update",
            args=[{"visible": visiblility_list.copy()},
                {"title": "Visualizing pose_" + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][trace_dict[f'pose_{i}']] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Pose: "},
        pad={"t": 0, "b": 10, "l": 10, "r": 10},
        steps=steps
    )]

    if ranges is None:
        pass
    else:
        assert ranges.shape == (3,2), f"{ranges.shape}"
        x_min, x_max = ranges[0,0].item(), ranges[0,1].item()
        y_min, y_max = ranges[1,0].item(), ranges[1,1].item()
        z_min, z_max = ranges[2,0].item(), ranges[2,1].item()

    x_center = (x_max+x_min)/2
    y_center = (y_max+y_min)/2
    z_center = (z_max+z_min)/2
    half_w = max(x_max-x_min, y_max-y_min, z_max-z_min)/2


    fig_sample.update_layout(
        scene=dict(xaxis=dict(range=[x_center-half_w, x_center+half_w]),
                   yaxis=dict(range=[y_center-half_w, y_center+half_w]),
                   zaxis=dict(range=[z_center-half_w, z_center+half_w]),
                   aspectmode='cube'),
        sliders=sliders,
        width=width,  # Adjust the width as needed
        height=height,  # Adjust the height as needed
        margin=dict(t=0, r=0, l=0, b=0)
    )


    fig_sample.data[trace_dict[f'pose_0']].visible = True

    return fig_grasp, fig_sample