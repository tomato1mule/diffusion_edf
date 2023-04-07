from typing import Tuple, Dict, List, Optional, Union

import torch
from diffusion_edf.data import DemoSeqDataset, DemoSequence, TargetPoseDemo, PointCloud, SE3
from diffusion_edf.pc_utils import get_plotly_fig

def visualize_pose(scene_pcd: PointCloud, grasp_pcd: PointCloud, poses: SE3, query: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, show_sample_points: bool = False):
    
    grasp_pl = grasp_pcd.plotly(point_size=1.0, name="grasp")
    grasp_geometry = [grasp_pl]
    if query is not None:
        query_points, query_attention = query
        query_opacity = query_attention ** 1
        query_pl = PointCloud.points_to_plotly(pcd=query_points, point_size=15.0, opacity=query_opacity / query_opacity.max())#, custom_data={'attention': query_attention.cpu()})
        grasp_geometry.append(query_pl)
    fig_grasp = get_plotly_fig("Grasp")
    fig_grasp = fig_grasp.add_traces(grasp_geometry)



    
    scene_pl = scene_pcd.plotly(point_size=1.0, name='scene')
    placement_geometry = [scene_pl]
    transformed_grasp_pcd = grasp_pcd.transformed(poses)
    for i in range(len(poses)):
        pose_pl = transformed_grasp_pcd[i].plotly(point_size=1.0, name=f'pose_{i}')
        placement_geometry.append(pose_pl)
    if show_sample_points:
        sample_pl = PointCloud.points_to_plotly(pcd=poses.points, point_size=7.0, colors=[0.2, 0.5, 0.8], name=f'sample_points')
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
        pad={"t": 50},
        steps=steps
    )]

    fig_sample.update_layout(
        sliders=sliders
    )

    fig_sample.data[trace_dict[f'pose_0']].visible = True

    return fig_grasp, fig_sample