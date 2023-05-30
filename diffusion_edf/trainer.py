import os
from typing import List, Tuple, Union, Optional, Dict, Callable
from datetime import datetime

from beartype import beartype
import yaml

import torch
from torch.utils.data import DataLoader

from diffusion_edf.data import DemoSeqDataset
from diffusion_edf import train_utils
from diffusion_edf.point_attentive_score_model import PointAttentiveScoreModel
from diffusion_edf.multiscale_score_model import MultiscaleScoreModel


class DiffusionEdfTrainer():
    configs_root_dir: str
    train_configs_file: str
    task_configs_file: str
    model_configs_file: str
    train_configs: Dict
    task_configs: Dict
    model_configs: Dict

    device: torch.device

    epochs: int
    steps: int
    log_dir: str

    @beartype
    def __init__(self, configs_root_dir: str,
                 train_configs_file: str,
                 task_configs_file: str):
        self.configs_root_dir = configs_root_dir
        self.train_configs_file = train_configs_file
        self.task_configs_file = task_configs_file
        with open(os.path.join(self.configs_root_dir, self.train_configs_file)) as file:
            self.train_configs = yaml.load(file, Loader=yaml.FullLoader)
        with open(os.path.join(self.configs_root_dir, self.task_configs_file)) as file:
            self.task_configs = yaml.load(file, Loader=yaml.FullLoader)
        self.model_configs_file = self.train_configs['model_config_file']
        with open(os.path.join(self.configs_root_dir, self.model_configs_file)) as file:
            self.model_configs = yaml.load(file, Loader=yaml.FullLoader)

        self.device = torch.device(self.train_configs['device'])
        self.max_epochs = self.train_configs['max_epochs']
        self.n_epochs_per_checkpoint = self.train_configs['n_epochs_per_checkpoint']
        self.n_samples_x_ref = self.train_configs['n_samples_x_ref']
        self.unit_length = 1/self.train_configs['rescale_factor']
        self.diffusion_schedules = self.train_configs['diffusion_configs']['time_schedules']
        self.n_schedules = len(self.diffusion_schedules)
        self.t_max = self.diffusion_schedules[0][0]
        for schedule in self.diffusion_schedules:
            assert schedule[0] > schedule[1]
            if schedule[0] > self.t_max:
                self.t_max = schedule[0]
        self.t_augment = self.train_configs['diffusion_configs']['t_augment']
            
        self.task_type = self.task_configs['task_type']
        self.contact_radius = self.task_configs['contact_radius']/self.unit_length

        self.trainloader: Optional[DataLoader] = None
        self.testloader: Optional[DataLoader] = None
        self.score_model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.logger: Optional[train_utils.LazyLogger] = None

    @property
    def is_initialized(self) -> bool:
        if self.trainloader is not None and self.score_model is not None and self.optimizer is not None and self.logger is not None:
            return True
        else:
            return False

    @beartype
    def get_current_time(self, postfix: Optional[str] = None) -> str:
        time_ = f"{datetime.now().strftime('%Y_%m_%d_%H-%M-%S')}"
        if postfix:
            time_ = time_ + '_' + postfix
        return time_
    
    @beartype
    def get_dataloader(self, dataset: DemoSeqDataset, 
                       n_batches: int,
                       shuffle: bool = True) -> DataLoader:
        proc_fn = train_utils.compose_proc_fn(self.train_configs['preprocess_config'])
        collate_fn = train_utils.get_collate_fn(task=self.task_type, proc_fn=proc_fn)
        dataloader = DataLoader(dataset, 
                                shuffle=shuffle, 
                                collate_fn=collate_fn, 
                                batch_size=n_batches)
        return dataloader
    
    @beartype
    def _init_dataloaders(self):
        trainset = DemoSeqDataset(dataset_dir=self.train_configs['trainset']['dataset_dir'], 
                                  annotation_file=self.train_configs['trainset']['annotation_file'], 
                                  device=self.device)
        self.trainloader = self.get_dataloader(dataset = trainset,
                                               shuffle = self.train_configs['trainset']['shuffle'],
                                               n_batches = self.train_configs['trainset']['n_batches'])
        if self.train_configs['testset'] is not None:
            testset = DemoSeqDataset(dataset_dir=self.train_configs['testset']['dataset_dir'], 
                                    annotation_file=self.train_configs['testset']['annotation_file'], 
                                    device=self.device)
            self.testloader = self.get_dataloader(dataset = testset,
                                                shuffle = self.train_configs['testset']['shuffle'],
                                                n_batches = self.train_configs['testset']['n_batches'])

    @beartype
    def get_model(self, deterministic: bool = False, 
                  device: Optional[Union[str, torch.device]] = None,
                  checkpoint_dir: Optional[str] = None,
                  ) -> Union[PointAttentiveScoreModel, MultiscaleScoreModel]:
        if device is None:
            device = self.device
        else:
            device = torch.device(device)

        if self.model_configs['model_name'] == 'PointAttentiveScoreModel':
            score_model =  PointAttentiveScoreModel(**self.model_configs['model_kwargs'], deterministic=deterministic)
        elif self.model_configs['model_name'] == 'MultiscaleScoreModel':
            score_model = MultiscaleScoreModel(**self.model_configs['model_kwargs'], device=device, deterministic=deterministic)
        else:
            raise ValueError(f"Unknown score model name: {self.model_configs['model_name']}")
        
        if checkpoint_dir is not None:
            checkpoint = torch.load(checkpoint_dir)
            score_model.load_state_dict(checkpoint['score_model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            steps = checkpoint['steps']
            print(f"Successfully Loaded checkpoint @ epoch: {epoch} (steps: {steps})")
        
        return score_model.to(device)
        
    @beartype
    def _init_model(self, deterministic: bool = False, 
                    device: Optional[Union[str, torch.device]] = None):
        self.score_model = self.get_model(deterministic=deterministic, device=device)

    @beartype
    def _init_optimizer(self):
        assert self.score_model is not None
        self.optimizer = torch.optim.Adam(list(self.score_model.parameters()), 
                                          **self.train_configs['optimizer_kwargs'])
    
    @beartype
    def _init_logging(self, log_name: str,
                      log_root_dir: Optional[str] = None,
                      resume_training: bool = False,
                      resume_checkpoint_dir: Optional[str] = None) -> bool:
        if log_root_dir is None:
            log_root_dir = self.train_configs['log_root_dir']

        if resume_training:
            raise NotImplementedError
            assert log_name is not None
            log_dir = os.path.join(log_root_dir, log_name)

            if resume_checkpoint_dir is None:
                resume_checkpoint_dir = sorted(os.listdir(os.path.join(log_dir, f'checkpoint')), key= lambda f:int(f.rstrip('.pt')))[-1]
            full_checkpoint_dir = os.path.join(log_dir, f'checkpoint', resume_checkpoint_dir)

            if input(f"Enter 'y' if you want to resume training from checkpoint: {full_checkpoint_dir}") == 'y':
                pass
            else:
                raise ValueError()
            
            checkpoint = torch.load(full_checkpoint_dir)
            self.score_model.load_state_dict(checkpoint['score_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            steps = checkpoint['steps']
            print(f"resume training from checkpoint: {full_checkpoint_dir}")

            self.epoch = epoch + 1
            self.steps = steps
            self.log_dir = log_dir
        else:
            assert resume_checkpoint_dir is None, f"Not resuming from checkpoint, but resume_checkpoint_dir is set to {resume_checkpoint_dir}"
            log_dir = os.path.join(log_root_dir, log_name)
            if os.path.exists(log_dir):
                raise ValueError(f'Directory "{log_dir}" already exists!')
            
            self.epoch = 0
            self.steps = 0
            self.log_dir = log_dir

        self.logger = train_utils.LazyLogger(log_dir=log_dir, 
                                             resume=resume_training,
                                             configs_root_dir=self.configs_root_dir)
        return True
    
    @beartype
    def init(self, log_name: str,
             log_root_dir: Optional[str] = None,
             resume_training: bool = False,
             resume_checkpoint_dir: Optional[str] = None,
             model: Optional[torch.nn.Module] = None) -> bool:
        if self.is_initialized:
            raise RuntimeError("Trainer already initialized!")
        
        self._init_dataloaders()
        if model is None:
            self._init_model()
        else:
            self.score_model = model
        self._init_optimizer()
        self._init_logging(log_name=log_name, 
                        log_root_dir=log_root_dir, 
                        resume_training=resume_training, 
                        resume_checkpoint_dir=resume_checkpoint_dir)
        return True








    