import os
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
from typing import List, Tuple, Optional, Union, Iterable, Dict, Any
import math
import argparse
import warnings

from beartype import beartype
import yaml
import torch

from edf_interface import data
from edf_interface.pyro import PyroServer, expose
from edf_interface.utils.manipulation_utils import compute_pre_pick_trajectories, compute_pre_place_trajectories
from diffusion_edf.agent import DiffusionEdfAgent

torch.set_printoptions(precision=4, sci_mode=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDF agents server for pick-and-place task')
    parser.add_argument('--configs-root-dir', type=str, help='')
    parser.add_argument('--server-name', type=str, default='agent', help='')
    parser.add_argument('--init-nameserver', action='store_true', help='')
    parser.add_argument('--compile-score-model-head', action='store_true', help='compile score head with torch.jit.script for faster inference, but may cause bug')
    parser.add_argument('--nameserver-host-ip', type=str, default='', help='')
    parser.add_argument('--nameserver-host-port', type=str, default='', help='')
    args = parser.parse_args()
    configs_root_dir = args.configs_root_dir
    server_name = args.server_name
    init_nameserver = args.init_nameserver
    compile_score_head = args.compile_score_model_head
    nameserver_host_ip = args.nameserver_host_ip
    nameserver_host_port = args.nameserver_host_port
    if not init_nameserver:
        init_nameserver = None
    if not nameserver_host_ip:
        nameserver_host_ip = None
    if not nameserver_host_port:
        nameserver_host_port = None
    else:
        nameserver_host_port = int(nameserver_host_port)


    # ---------------------------------------------------------------------------- #
    # Initialize Pyro Server
    # ---------------------------------------------------------------------------- #
    server = PyroServer(server_name='agent', init_nameserver=init_nameserver, 
                        nameserver_host=nameserver_host_ip, nameserver_port=nameserver_host_port)


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

    server_configs_dir = os.path.join(configs_root_dir, 'server.yaml')
    with open(server_configs_dir) as f:
        server_configs = yaml.load(f, Loader=yaml.FullLoader)
        
    pick_agent = DiffusionEdfAgent(
        model_kwargs_list=agent_configs['model_kwargs'][f"pick_models_kwargs"],
        preprocess_config=preprocess_config,
        unprocess_config=unprocess_config,
        device=device,
        compile_score_head=compile_score_head,
        critic_kwargs=agent_configs['model_kwargs'].get(f"pick_critic_kwargs", None)
    )

    place_agent = DiffusionEdfAgent(
        model_kwargs_list=agent_configs['model_kwargs'][f"place_models_kwargs"],
        preprocess_config=preprocess_config,
        unprocess_config=unprocess_config,
        device=device,
        compile_score_head=compile_score_head,
        critic_kwargs=agent_configs['model_kwargs'].get(f"place_critic_kwargs", None)
    )

    @beartype
    class AgentService():
        def __init__(self, configs: Dict[str, Any]):
            self.pick_diffusion_configs: Dict[str, Any] = configs['pick_diffusion_configs']
            self.pick_trajectory_configs: Dict[str, Any] = configs['pick_trajectory_configs']
            self.place_diffusion_configs: Dict[str, Any] = configs['place_diffusion_configs']
            self.place_trajectory_configs: Dict[str, Any] = configs['place_trajectory_configs']
            self.reconfigurable_configs = ['pick_diffusion_configs', 
                                           'pick_trajectory_configs', 
                                           'place_diffusion_configs', 
                                           'place_trajectory_configs']
            for config_name in self.reconfigurable_configs:
                assert hasattr(self, config_name)
        
        def _reconfigure(self, name: str, value: Dict[str, Any]):
            if name not in self.reconfigurable_configs:
                raise ValueError(f"'{name}' not in reconfiguratible configs ({self.reconfigurable_configs})")
            else:
                setattr(self, name, value)

        @expose
        def reconfigure(self, name: str, value: Dict[str, Any]) -> bool:
            self._reconfigure(name=name, value=value)

        @expose
        def get_configs(self) -> Dict[str, Any]:
            output = {}
            for config_name in self.reconfigurable_configs:
                output[config_name] = getattr(self, config_name)
            return output

        def compute_pick_trajectory(self, pick_poses: data.SE3) -> List[data.SE3]:
            trajectories = compute_pre_pick_trajectories(
                pick_poses=pick_poses, 
                **self.pick_trajectory_configs
            )

            return trajectories
        
        def compute_place_trajectory(self, place_poses: data.SE3,
                                     scene_pcd: data.PointCloud, 
                                     grasp_pcd: data.PointCloud) -> List[data.SE3]:
            trajectories = compute_pre_place_trajectories(
                place_poses=place_poses, 
                scene_pcd=scene_pcd,
                grasp_pcd=grasp_pcd,
                **self.place_trajectory_configs
            )

            return trajectories


        def _denoise(self, scene_pcd: data.PointCloud, 
                     grasp_pcd: data.PointCloud,
                     current_poses: data.SE3,
                     task_name: str,
                     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            
            assert current_poses.poses.ndim == 2 and current_poses.poses.shape[-1] == 7, f"{current_poses.shape}"
            n_init_poses = len(current_poses)
            
            if task_name == 'pick':
                Ts, scene_proc, grasp_proc, info = pick_agent.sample(
                    scene_pcd=scene_pcd.to(device), 
                    grasp_pcd=grasp_pcd.to(device), 
                    Ts_init=current_poses.to(device),
                    N_steps_list=self.pick_diffusion_configs['N_steps_list'], 
                    timesteps_list=self.pick_diffusion_configs['timesteps_list'], 
                    temperatures_list=self.pick_diffusion_configs['temperatures_list'],
                    diffusion_schedules_list=self.pick_diffusion_configs['diffusion_schedules_list'],
                    log_t_schedule=self.pick_diffusion_configs['log_t_schedule'],
                    time_exponent_temp=self.pick_diffusion_configs['time_exponent_temp'],
                    time_exponent_alpha=self.pick_diffusion_configs['time_exponent_alpha'],
                    return_info=True
                )

                assert Ts.ndim == 3 and Ts.shape[-2] == n_init_poses and Ts.shape[-1] == 7, f"{Ts.shape}"

            elif task_name == 'place':
                Ts, scene_proc, grasp_proc, info = place_agent.sample(
                    scene_pcd=scene_pcd.to(device), 
                    grasp_pcd=grasp_pcd.to(device), 
                    Ts_init=current_poses.to(device),
                    N_steps_list=self.place_diffusion_configs['N_steps_list'], 
                    timesteps_list=self.place_diffusion_configs['timesteps_list'], 
                    temperatures_list=self.place_diffusion_configs['temperatures_list'],
                    diffusion_schedules_list=self.place_diffusion_configs['diffusion_schedules_list'],
                    log_t_schedule=self.place_diffusion_configs['log_t_schedule'],
                    time_exponent_temp=self.place_diffusion_configs['time_exponent_temp'],
                    time_exponent_alpha=self.place_diffusion_configs['time_exponent_alpha'],
                    return_info=True
                )

                assert Ts.ndim == 3 and Ts.shape[-2] == n_init_poses and Ts.shape[-1] == 7, f"{Ts.shape}"
            else:
                raise ValueError(f"Unknown task name '{task_name}'")

            return Ts, info
        
        @expose
        def denoise(self, scene_pcd: data.PointCloud, 
                     grasp_pcd: data.PointCloud,
                     current_poses: data.SE3,
                     task_name: str,
                     ) -> Tuple[List[data.SE3], Dict[str, Any]]:
            traj_tensors, info = self._denoise(scene_pcd=scene_pcd, grasp_pcd=grasp_pcd, current_poses=current_poses, task_name=task_name)
            traj_tensors = traj_tensors.detach().cpu()
            trajectories = []
            for i in range(traj_tensors.shape[-2]):
                Ts = traj_tensors[:,i]
                Ts = data.SE3(poses=Ts)

                if task_name == 'pick':
                    unproc_fn = pick_agent.unprocess_fn
                    Ts = unproc_fn(Ts)
                elif task_name == 'place':
                    unproc_fn = place_agent.unprocess_fn
                    Ts = unproc_fn(Ts)
                else:
                    raise ValueError(f"Unknown task name: '{task_name}'")
                
                trajectories.append(Ts)
                

            # info = {}
            def recursive_cuda_to_cpu(info: Dict):
                for k,v in info.items():
                    if isinstance(v, torch.Tensor):
                        info[k]=v.to('cpu')
                    elif isinstance(v, Dict):
                        info[k]=recursive_cuda_to_cpu(v)
            recursive_cuda_to_cpu(info)
            
            return Ts, info
            
            
            
        
        @expose
        def request_trajectories(self, scene_pcd: data.PointCloud, 
                                 grasp_pcd: data.PointCloud,
                                 current_poses: data.SE3,
                                 task_name: str,
                                 ) -> Tuple[List[data.SE3], Dict[str, Any]]:
            denoise_seq, info = self._denoise(
                scene_pcd=scene_pcd, grasp_pcd=grasp_pcd, current_poses=current_poses, task_name=task_name
            ) # (n_time, n_init_pose, 7)
            denoise_seq = denoise_seq.to(device='cpu')

            Ts = data.SE3(poses=denoise_seq[-1])

            if task_name == 'pick':
                unproc_fn = pick_agent.unprocess_fn
                Ts = unproc_fn(Ts)
                trajectories = self.compute_pick_trajectory(pick_poses=Ts)
            elif task_name == 'place':
                unproc_fn = place_agent.unprocess_fn
                Ts = unproc_fn(Ts)
                trajectories = self.compute_place_trajectory(place_poses=Ts, scene_pcd=scene_pcd, grasp_pcd=grasp_pcd)
            else:
                raise ValueError(f"Unknown task name: '{task_name}'")

            # info = {}
            def recursive_cuda_to_cpu(info: Dict):
                for k,v in info.items():
                    if isinstance(v, torch.Tensor):
                        info[k]=v.to('cpu')
                    elif isinstance(v, Dict):
                        info[k]=recursive_cuda_to_cpu(v)
            recursive_cuda_to_cpu(info)

            return trajectories, info
        

    server.register_service(service=AgentService(configs=server_configs))
    server.run(nonblocking=False)

    server.close()


