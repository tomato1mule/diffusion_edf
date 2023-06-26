import os
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
from typing import List, Tuple, Optional, Union, Iterable
import math
import argparse
import warnings

from beartype import beartype
import yaml
import torch

from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroServer, expose
from diffusion_edf.agent import DiffusionEdfAgent

torch.set_printoptions(precision=4, sci_mode=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDF agents server for pick-and-place task')
    parser.add_argument('--configs-root-dir', type=str, help='')
    parser.add_argument('--server-name', type=str, default='agent', help='')
    parser.add_argument('--init-nameserver', action='store_true', help='')
    args = parser.parse_args()
    configs_root_dir = args.configs_root_dir
    server_name = args.server_name
    init_nameserver = args.init_nameserver
    if not init_nameserver:
        init_nameserver = None


    # ---------------------------------------------------------------------------- #
    # Initialize Pyro Server
    # ---------------------------------------------------------------------------- #
    server = PyroServer(server_name='agent', init_nameserver=init_nameserver)


    # ---------------------------------------------------------------------------- #
    # Initialize Models
    # ---------------------------------------------------------------------------- #
    agent_configs_dir = os.path.join(configs_root_dir, 'agent.yaml')
    with open(agent_configs_dir) as f:
        agent_configs = yaml.load(f, Loader=yaml.FullLoader)
    device = agent_configs['device']

    preprocess_configs_dir = os.path.join(configs_root_dir, 'preprocess.yaml')
    with open(preprocess_configs_dir) as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
        unprocess_config = preprocess_config['unprocess_config']
        preprocess_config = preprocess_config['preprocess_config']

    pick_agent = DiffusionEdfAgent(
        model_kwargs_list=agent_configs['model_kwargs'][f"pick_models_kwargs"],
        preprocess_config=preprocess_config,
        unprocess_config=unprocess_config,
        device=device
    )

    place_agent = DiffusionEdfAgent(
        model_kwargs_list=agent_configs['model_kwargs'][f"place_models_kwargs"],
        preprocess_config=preprocess_config,
        unprocess_config=unprocess_config,
        device=device
    )

    @beartype
    class AgentService():
        def __init__(self):
            pass

        @expose
        def infer_target_poses(self, scene_pcd: PointCloud, 
                               task_name: str,
                               grasp_pcd: PointCloud,
                               current_poses: SE3,
                               N_steps_list: List[List[int]],
                               timesteps_list: List[List[float]],
                               temperature_list: List[float],
                               return_full_trajectory: bool = False,
                               ) -> List[SE3]:
            
            assert current_poses.poses.ndim == 2 and current_poses.poses.shape[-1] == 7, f"{current_poses.shape}"
            n_init_poses = len(current_poses)
            
            if task_name == 'pick':
                Ts, scene_proc, grasp_proc = pick_agent.sample(
                    scene_pcd=scene_pcd.to(device), 
                    grasp_pcd=grasp_pcd.to(device), 
                    Ts_init=current_poses.to(device),
                    N_steps_list=N_steps_list, 
                    timesteps_list=timesteps_list, 
                    temperature_list=temperature_list
                )

                assert Ts.ndim == 3 and Ts.shape[-2] == n_init_poses and Ts.shape[-1] == 7, f"{Ts.shape}"
                Ts = Ts.to('cpu')

                Ts_out: List[SE3] = []
                for i in range(n_init_poses):
                    Ts_out.append(
                        pick_agent.unprocess_fn(
                            SE3(poses = Ts[:, i, :]) if return_full_trajectory else SE3(poses = Ts[-1, i, :])
                        )
                    )
            elif task_name == 'place':
                Ts, scene_proc, grasp_proc = place_agent.sample(
                    scene_pcd=scene_pcd.to(device), 
                    grasp_pcd=grasp_pcd.to(device), 
                    Ts_init=current_poses.to(device),
                    N_steps_list=N_steps_list, 
                    timesteps_list=timesteps_list, 
                    temperature_list=temperature_list
                )

                assert Ts.ndim == 3 and Ts.shape[-2] == n_init_poses and Ts.shape[-1] == 7, f"{Ts.shape}"
                Ts = Ts.to('cpu')
                
                Ts_out: List[SE3] = []
                for i in range(n_init_poses):
                    Ts_out.append(
                        place_agent.unprocess_fn(
                            SE3(poses = Ts[:, i, :]) if return_full_trajectory else SE3(poses = Ts[-1, i, :])
                        )
                    )
            else:
                raise ValueError(f"Unknown task name '{task_name}'")

            return Ts_out
        

    server.register_service(service=AgentService())
    server.run(nonblocking=False)

    server.close()


