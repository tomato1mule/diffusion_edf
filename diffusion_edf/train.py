import os
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
from typing import List, Tuple, Optional, Union, Iterable
import math
import argparse

import torch

from diffusion_edf import train_utils
from diffusion_edf.trainer import DiffusionEdfTrainer

torch.set_printoptions(precision=4, sci_mode=False)

def train_score_model(configs_root_dir: str, 
                      train_configs_file: str, 
                      task_configs_file: str, 
                      resume_training: bool = False,
                      log_name: Optional[str] = None,
                      log_name_postfix: Optional[str] = None):
    if not configs_root_dir:
        raise ValueError(f"configs_root_dir must be specified.")
    if resume_training and not log_name:
        raise ValueError(f"log_name must be specified if resume_training is True.")
    
    trainer = DiffusionEdfTrainer(configs_root_dir=configs_root_dir,
                                  train_configs_file=train_configs_file,
                                  task_configs_file=task_configs_file)
    if not log_name and not resume_training:
        log_name = trainer.get_current_time(postfix=log_name_postfix)

    init_epoch = trainer.init(
        log_name = log_name,
        resume_training = resume_training,
    )

    for epoch in range(init_epoch, trainer.max_epochs+1):
        for n, demo_batch in enumerate(trainer.trainloader):
            B = len(demo_batch)
            assert B == 1, "Batch training is not supported yet."

            scene_input, grasp_input, T_target = train_utils.flatten_batch(demo_batch=demo_batch) # T_target: (Nbatch, Ngrasps, 7)
            T_target = T_target.squeeze(0) # (B=1, N_poses=1, 7) -> (1,7) 

            save_checkpoint = (epoch % trainer.n_epochs_per_checkpoint == 0) and n == len(trainer.trainloader)-1
            trainer.train_once(
                T_target=T_target,
                scene_input=scene_input,
                grasp_input=grasp_input,
                epoch=epoch,
                save_checkpoint = save_checkpoint,
                checkpoint_count = epoch // trainer.n_epochs_per_checkpoint
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate EDF agents for pick-and-place task')
    parser.add_argument('--configs-root-dir', type=str,
                        help='')
    parser.add_argument('--train-configs-file', type=str, default='train_configs.yaml',
                        help='')
    parser.add_argument('--task-configs-file', type=str, default='task_configs.yaml',
                        help='')
    parser.add_argument('--log-name', type=str,
                        help='')
    parser.add_argument('--log-name-postfix', type=str,
                        help='')
    parser.add_argument('--resume-training', action='store_true',
                    help='')
    args = parser.parse_args()

    configs_root_dir = args.configs_root_dir
    train_configs_file = args.train_configs_file
    task_configs_file = args.task_configs_file
    log_name = args.log_name
    log_name_postfix = args.log_name_postfix
    resume_training = args.resume_training    

    train_score_model(configs_root_dir=configs_root_dir,
                      train_configs_file=train_configs_file,
                      task_configs_file=task_configs_file,
                      log_name=log_name,
                      log_name_postfix=log_name_postfix,
                      resume_training=resume_training
                      )