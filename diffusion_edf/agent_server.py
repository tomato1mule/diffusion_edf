import os
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
from typing import List, Tuple, Optional, Union, Iterable
import math
import argparse
import warnings

from beartype import beartype
import yaml
import torch

from edf_interface.agent_server import AgentHandleAbstractBase, AgentServer
from edf_interface.data import SE3, PointCloud
from diffusion_edf.trainer import DiffusionEdfTrainer
from diffusion_edf.train_utils import proc_fn

torch.set_printoptions(precision=4, sci_mode=False)


@beartype
def get_models(configs_root_dir: str, 
               train_configs_file: str, 
               task_configs_file: str, 
               checkpoint_dir: str,
               device: str,
               n_warmups: int = 10,
               ):

    trainer = DiffusionEdfTrainer(
        configs_root_dir=configs_root_dir,
        train_configs_file=train_configs_file,
        task_configs_file=task_configs_file,
        device=device
    )

    trainer._init_dataloaders()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The TorchScript type system doesn*')
        
        model = trainer.get_model(
            checkpoint_dir=checkpoint_dir,
            deterministic=False, 
            device = device
        ).eval()

    if n_warmups:
        trainer.warmup_score_model(
            score_model = model, 
            n_warmups=10
        )
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDF agents server for pick-and-place task')
    parser.add_argument('--configs-dir', type=str, help='')
    args = parser.parse_args()

    configs_dir = args.configs_dir
    with open(os.path.join(configs_dir, 'agent.yaml'), 'r') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    device = model_configs['device']

    pick_models = []
    for kwargs in model_configs['pick_models_kwargs']:
        pick_models.append(get_models(**kwargs, device=device))

    place_models = []
    for kwargs in model_configs['place_models_kwargs']:
        place_models.append(get_models(**kwargs, device=device))


    @beartype
    class AgentHandle(AgentHandleAbstractBase):
        def __init__(self):
            pass

        def infer_target_poses(self, scene_pcd: PointCloud, 
                            task_name: str,
                            grasp_pcd: Optional[PointCloud] = None,
                            current_poses: Optional[SE3] = None) -> SE3:
            
            return SE3(poses=torch.tensor([[1., 0., 0., 0., 0., 0., 0.,]]))

    



