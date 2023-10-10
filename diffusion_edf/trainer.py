import os, sys
from typing import List, Tuple, Union, Optional, Dict, Callable
from datetime import datetime
import warnings

from tqdm import tqdm
from beartype import beartype
import yaml

import torch
from torch.utils.data import DataLoader

from edf_interface.data import PointCloud, SE3, DemoDataset
from diffusion_edf.gnn_data import FeaturedPoints
from diffusion_edf import train_utils
from diffusion_edf.score_model_base import ScoreModelBase
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
    steps: int
    log_dir: str

    @beartype
    def __init__(self, configs_root_dir: str,
                 train_configs_file: str,
                 task_configs_file: str,
                 device: Optional[Union[str, torch.device]] = None):
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

        if device is None:
            self.device = torch.device(self.train_configs['device'])
        else:
            self.device = torch.device(device)
        self.max_epochs = self.train_configs['max_epochs']
        self.n_epochs_per_checkpoint = self.train_configs['n_epochs_per_checkpoint']
        self.n_samples_x_ref = self.train_configs['n_samples_x_ref']
        self.unit_length = 1/self.train_configs['rescale_factor']
        self.diffusion_schedules = self.train_configs['diffusion_configs']['time_schedules']
        self.diffusion_xref_bbox = self.train_configs['diffusion_configs'].get('diffusion_xref_bbox', None)
        if self.diffusion_xref_bbox is not None:
            self.diffusion_xref_bbox = torch.tensor(self.diffusion_xref_bbox)
        self.n_schedules = len(self.diffusion_schedules)
        self.t_max = self.diffusion_schedules[0][0]
        for schedule in self.diffusion_schedules:
            assert schedule[0] >= schedule[1]
            if schedule[0] >= self.t_max:
                self.t_max = schedule[0]
        self.t_augment = self.train_configs['diffusion_configs']['t_augment']
            
        self.task_type = self.task_configs['task_type']
        self.contact_radius = self.task_configs['contact_radius']/self.unit_length

        self.trainloader: Optional[DataLoader] = None
        self.testloader: Optional[DataLoader] = None
        self.score_model: Optional[ScoreModelBase] = None
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
    def get_dataloader(self, dataset: DemoDataset, 
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
    def _init_dataloaders(self, half_precision: bool = False):
        trainset = DemoDataset(dataset_dir=self.train_configs['trainset']['dataset_dir'], 
                               annotation_file=self.train_configs['trainset']['annotation_file'], 
                               device=self.device,
                               dtype = torch.float16 if half_precision else torch.float32)
        self.trainloader = self.get_dataloader(dataset = trainset,
                                               shuffle = self.train_configs['trainset']['shuffle'],
                                               n_batches = self.train_configs['trainset']['n_batches'])
        if self.train_configs['testset'] is not None:
            testset = DemoDataset(dataset_dir=self.train_configs['testset']['dataset_dir'], 
                                  annotation_file=self.train_configs['testset']['annotation_file'], 
                                  device=self.device,
                                  dtype = torch.float16 if half_precision else torch.float32)
            self.testloader = self.get_dataloader(dataset = testset,
                                                  shuffle = self.train_configs['testset']['shuffle'],
                                                  n_batches = self.train_configs['testset']['n_batches'])

    @beartype
    def get_model(self, deterministic: bool = False, 
                  device: Optional[Union[str, torch.device]] = None,
                  checkpoint_dir: Optional[str] = None,
                  strict: bool = True
                  ) -> ScoreModelBase:
        if device is None:
            device = self.device
        else:
            device = torch.device(device)

        if self.model_configs['model_name'] == 'PointAttentiveScoreModel':
            score_model =  PointAttentiveScoreModel(**self.model_configs['model_kwargs'], deterministic=deterministic)
        elif self.model_configs['model_name'] == 'MultiscaleScoreModel':
            score_model = MultiscaleScoreModel(**self.model_configs['model_kwargs'], deterministic=deterministic)
        else:
            raise ValueError(f"Unknown score model name: {self.model_configs['model_name']}")
        
        if checkpoint_dir is not None:
            checkpoint = torch.load(checkpoint_dir)
            score_model.load_state_dict(checkpoint['score_model_state_dict'], strict=strict)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=strict)
            epoch = checkpoint['epoch']
            steps = checkpoint['steps']
            print(f"Successfully Loaded checkpoint @ epoch: {epoch} (steps: {steps})")
        
        return score_model.to(device)
            
    @beartype
    def _init_model(self, deterministic: bool = False, 
                    device: Optional[Union[str, torch.device]] = None):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The TorchScript type system doesn*')
            warnings.filterwarnings('ignore', message='Multiscale Tensor Field: zero edges detected!')
            
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
                      resume_checkpoint_dir: Optional[str] = None) -> int:
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

            init_epoch = epoch + 1
            self.steps = steps
            self.log_dir = log_dir
        else:
            assert resume_checkpoint_dir is None, f"Not resuming from checkpoint, but resume_checkpoint_dir is set to {resume_checkpoint_dir}"
            log_dir = os.path.join(log_root_dir, log_name)
            if os.path.exists(log_dir):
                raise ValueError(f'Directory "{log_dir}" already exists!')
            
            init_epoch = 0
            self.steps = 0
            self.log_dir = log_dir

        self.logger = train_utils.LazyLogger(log_dir=log_dir, 
                                             resume=resume_training,
                                             configs_root_dir=self.configs_root_dir)
        return init_epoch
    
    @beartype
    def init(self, log_name: str,
             log_root_dir: Optional[str] = None,
             resume_training: bool = False,
             resume_checkpoint_dir: Optional[str] = None,
             model: Optional[ScoreModelBase] = None) -> int:
        if self.is_initialized:
            raise RuntimeError("Trainer already initialized!")
        
        self._init_dataloaders()
        if model is None:
            self._init_model()
        else:
            self.score_model = model
        self._init_optimizer()
        init_epoch = self._init_logging(
            log_name=log_name, 
            log_root_dir=log_root_dir, 
            resume_training=resume_training, 
            resume_checkpoint_dir=resume_checkpoint_dir
        )
        return init_epoch
    
    @beartype
    def save(self, epoch: int):
        torch.save({'epoch': epoch,
                    'steps': self.steps,
                    'score_model_state_dict': self.score_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.log_dir, f'checkpoint/{epoch}.pt'))
        
        print(f"(Epoch: {epoch}) Successfully saved logs to: {self.log_dir}")

    @beartype
    def biequiv_diffusion(self, T_init: torch.Tensor, time: Union[float, torch.Tensor],
                          scene_points: FeaturedPoints, grasp_points: FeaturedPoints,
                          ang_mult: Union[int, float],
                          lin_mult: Union[int, float],
                          n_samples_x_ref: int,
                          contact_radius: Optional[Union[int, float]] = None,
                          xref_bbox: Optional[torch.Tensor] = None,
                          ) -> Tuple[torch.Tensor, 
                                     torch.Tensor, 
                                     torch.Tensor,
                                     Tuple[torch.Tensor, torch.Tensor],
                                     Tuple[torch.Tensor, torch.Tensor]
                                     ]:
        """
        Input Shapes:
            T_init: (nT, 7);  currently only nT=1 is implemented.
            time: (nT,)
        Output Shapes:
            T_diffused: (nT * n_samples_x_ref, 7)
            delta_T: (nT * n_samples_x_ref, 7)
            time_in: (nT * n_samples_x_ref, )
            gt_<...>_score: (nT * n_samples_x_ref, 3)
        """
        assert T_init.ndim == 2 and T_init.shape[-1] == 7, f"T_init.shape must be (N_poses, 7), but {T_init.shape} is given."
        nT = len(T_init)
        if nT != 1:
            raise NotImplementedError(f"T_init.shape = (nT,7) with nT > 1 is not yet implemented, but {T_init.shape} is given.")

        if isinstance(time, float):
            time = torch.tensor([time], device=T_init.device).expand(nT)
        assert time.shape == (nT,)

        ang_mult, lin_mult = float(ang_mult), float(lin_mult)
        if contact_radius is None:
            contact_radius = self.contact_radius
        else:
            contact_radius = float(contact_radius)
            
        if self.diffusion_xref_bbox is not None:
            if grasp_points.x.dtype != self.diffusion_xref_bbox.dtype or grasp_points.x.device != self.diffusion_xref_bbox.device:
                self.diffusion_xref_bbox = self.diffusion_xref_bbox.to(dtype=grasp_points.x.dtype, device=grasp_points.x.device)
            
        x_ref, n_neighbors = train_utils.transform_and_sample_reference_points(
            T_target=T_init,
            scene_points=scene_points,
            grasp_points=grasp_points,
            contact_radius=contact_radius,
            n_samples_x_ref=n_samples_x_ref,
            xref_bbox=xref_bbox
        )
        T_diffused, delta_T, time_in, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref) = train_utils.diffuse_T_target(
            T_target=T_init, 
            x_ref=x_ref, 
            time=time, 
            lin_mult=lin_mult,
            ang_mult=ang_mult
        )

        return T_diffused, delta_T, time_in, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref)
    
    @beartype
    def train_once(self, T_target: torch.Tensor, 
                   scene_input: FeaturedPoints, 
                   grasp_input: FeaturedPoints,
                   epoch: int,
                   save_checkpoint: bool,
                   checkpoint_count: Optional[int] = None):
        if scene_input.b.max() != 0:
            raise NotImplementedError(f"Batched training is currently not supported. (n_batch: {scene_input.b})")
        assert self.is_initialized, f"Trainer not initialized!"
        assert T_target.shape == (1,7), f"T_target.shape must be (1,7), but {T_target.shape} is given."

        self.optimizer.zero_grad(set_to_none=True)

        loss, T_diffused, fp_info, tensor_info, statistics = self.run_once(T_target=T_target, 
                                                                           scene_input=scene_input, 
                                                                           grasp_input=grasp_input)
        loss.backward()
        self.optimizer.step()
        self.steps += 1

        ### Record scalars ###
        with torch.no_grad():
            for tag, scalar_value in statistics.items():
                self.logger.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.steps)
        
        ### Record 3d points ###
        if save_checkpoint:
            assert isinstance(checkpoint_count, int)
            self.record_pcd(
                T_target=T_target.detach(), 
                T_diffused=T_diffused.detach(), 
                scene_input=scene_input, 
                grasp_input=grasp_input,
                scene_output=fp_info['key_fp'],
                grasp_output=fp_info['query_fp'],
                count=checkpoint_count
            )

            self.save(epoch=epoch)
            

    @beartype
    def run_once(self, T_target: torch.Tensor, 
                 scene_input: FeaturedPoints, 
                 grasp_input: FeaturedPoints) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict, Dict]:
        if scene_input.b.max() != 0:
            raise NotImplementedError(f"Batched training is currently not supported. (n_batch: {scene_input.b})")
        assert self.is_initialized, f"Trainer not initialized!"
        assert T_target.shape == (1,7), f"T_target.shape must be (1,7), but {T_target.shape} is given."
        
        ########################################## Augmentation #########################################
        if self.t_augment is not None:
            T_target, _, __, ___, ____ = self.biequiv_diffusion(
                T_init=T_target, 
                time=self.t_augment,
                scene_points=scene_input,
                grasp_points=grasp_input,
                ang_mult=self.score_model.ang_mult,
                lin_mult=self.score_model.lin_mult,
                n_samples_x_ref=1,
            )
        ##################################################################################################

        ############################################ Diffusion ###########################################
        time_in = torch.empty(0, device=self.device)
        T_diffused = torch.empty(0,7, device=self.device)
        gt_ang_score, gt_lin_score = torch.empty(0,3, device=self.device), torch.empty(0,3, device=self.device)
        gt_ang_score_ref, gt_lin_score_ref = torch.empty(0,3, device=self.device), torch.empty(0,3, device=self.device)
        
        for time_schedule in self.diffusion_schedules:
            time = train_utils.random_time(
                min_time=time_schedule[1], 
                max_time=time_schedule[0], 
                device=T_target.device
            ) # Shape: (1,)
            
            T_diffused_, delta_T_, time_in_, gt_score_, gt_score_ref_ = self.biequiv_diffusion(
                T_init=T_target, 
                time=time,
                scene_points=scene_input,
                grasp_points=grasp_input,
                ang_mult=self.score_model.ang_mult,
                lin_mult=self.score_model.lin_mult,
                n_samples_x_ref=self.n_samples_x_ref
            )            
            
            (gt_ang_score_, gt_lin_score_), (gt_ang_score_ref_, gt_lin_score_ref_) = gt_score_, gt_score_ref_
            T_diffused = torch.cat([T_diffused, T_diffused_], dim=0)
            time_in = torch.cat([time_in, time_in_], dim=0)
            gt_ang_score = torch.cat([gt_ang_score, gt_ang_score_], dim=0)
            gt_lin_score = torch.cat([gt_lin_score, gt_lin_score_], dim=0)
            gt_ang_score_ref = torch.cat([gt_ang_score_ref, gt_ang_score_ref_], dim=0)
            gt_lin_score_ref = torch.cat([gt_lin_score_ref, gt_lin_score_ref_], dim=0)

        ##################################################################################################

        loss, fp_info, tensor_info, statistics = self.score_model.get_train_loss(Ts=T_diffused, time=time_in, key_pcd=scene_input, query_pcd=grasp_input,
                                                                                 target_ang_score=gt_ang_score, target_lin_score=gt_lin_score)
        
        return loss, T_diffused, fp_info, tensor_info, statistics


    #@beartype
    def record_pcd(self, T_target: torch.Tensor,
                   T_diffused: torch.Tensor,
                   scene_input: FeaturedPoints, 
                   grasp_input: FeaturedPoints,
                   scene_output: Optional[FeaturedPoints],
                   grasp_output: FeaturedPoints,
                   count: int,
                   ):
        with torch.no_grad():
            scene_pcd = PointCloud(points=scene_input.x, colors=scene_input.f)
            grasp_pcd = PointCloud(points=grasp_input.x, colors=grasp_input.f)
            target_pose_pcd = PointCloud.merge(
                scene_pcd,
                grasp_pcd.transformed(SE3(T_target), squeeze=True),
            )
            diffused_pose_pcd = PointCloud.merge(
                scene_pcd,
                grasp_pcd.transformed(SE3(T_diffused))[0],
            )
            if scene_output is not None:
                scene_attn_pcd = PointCloud(points=scene_output.x.detach().cpu(), 
                                            colors=scene_output.w.detach().cpu(),
                                            cmap='magma')
            else:
                scene_attn_pcd = None
            grasp_attn_pcd = PointCloud(points=grasp_output.x.detach().cpu(), 
                                        colors=grasp_output.w.detach().cpu(),
                                        cmap='magma')
        
            query_weight, query_points, query_point_batch = grasp_output.w.detach(), grasp_output.x.detach(), grasp_output.b.detach(), 
            batch_vis_idx = (query_point_batch == 0).nonzero().squeeze(-1)
            query_weight, query_points = query_weight[batch_vis_idx], query_points[batch_vis_idx]

            N_repeat = 500
            query_points_colors = torch.tensor([0.01, 1., 1.], device=query_weight.device, dtype=query_weight.dtype).expand(N_repeat, 1, 3) * query_weight[None, :, None]
            r_query_ball = 0.5

            ball = torch.randn(N_repeat,1,3, device=query_points.device, dtype=query_points.dtype)
            ball = ball/ball.norm(dim=-1, keepdim=True) * r_query_ball
            query_points = (query_points + ball).reshape(-1,3)
            query_points_colors = query_points_colors.reshape(-1,3)


        if scene_attn_pcd is not None:
            self.logger.add_3d(
                tag = "Scene Attention",
                data = {
                    "vertex_positions": scene_attn_pcd.points.cpu(),
                    "vertex_colors": scene_attn_pcd.colors.cpu(),  # (N, 3)
                },
                step=count,
            )

        self.logger.add_3d(
            tag = "Grasp Attention",
            data = {
                # "vertex_positions": query_points.repeat(max(int(1000//len(query_points)),1),1).cpu(),      # There is a bug with too small number of points so repeat
                # "vertex_colors": query_points_colors.repeat(max(int(1000//len(query_points)),1),1).cpu(),  # (N, 3)
                "vertex_positions": query_points.cpu(),      # There is a bug with too small number of points so repeat
                "vertex_colors": query_points_colors.cpu(),  # (N, 3)
            },
            step=count,
        )

        self.logger.add_3d(
            tag = "Target Pose",
            data = {
                "vertex_positions": target_pose_pcd.points.cpu(),
                "vertex_colors": target_pose_pcd.colors.cpu(),  # (N, 3)
            },
            step=count,
        )

        self.logger.add_3d(
            tag = "Diffused Pose",
            data = {
                "vertex_positions": diffused_pose_pcd.points.cpu(),
                "vertex_colors": diffused_pose_pcd.colors.cpu(),  # (N, 3)
            },
            step=count,
            #description=f"Diffuse time: {time_in[0].item()} || eps: {eps.item()} || std: {std.item()}",
        )

        self.logger.add_3d(
            tag = "Grasp",
            data = {
                "vertex_positions": grasp_pcd.points.cpu(),
                "vertex_colors": grasp_pcd.colors.cpu(),  # (N, 3)
            },
            step=count,
        )


    def warmup_score_model(self, score_model: ScoreModelBase, n_warmups: int = 10):
        assert self.trainloader is not None
        score_model.requires_grad_(False)

        for iters in tqdm(range(n_warmups), file=sys.stdout):
            demo_batch = next(iter(self.trainloader))

            B = len(demo_batch)
            assert B == 1, "Batch training is not supported yet."

            scene_input, grasp_input, T_target = train_utils.flatten_batch(demo_batch=demo_batch) # T_target: (Nbatch, Ngrasps, 7)
            T_target = T_target.squeeze(0) # (B=1, N_poses=1, 7) -> (1,7) 

            key_pcd_multiscale: List[FeaturedPoints] = score_model.get_key_pcd_multiscale(scene_input)
            query_pcd: FeaturedPoints = score_model.get_query_pcd(grasp_input)
                
            for time_schedule in self.diffusion_schedules:
                time = train_utils.random_time(
                    min_time=time_schedule[1], 
                    max_time=time_schedule[0], 
                    device=T_target.device,
                    dtype=T_target.dtype
                ) # Shape: (1,)

                with torch.no_grad():
                    T, _, __, ___, ____ = self.biequiv_diffusion(
                        T_init=T_target, 
                        time=time,
                        scene_points=scene_input,
                        grasp_points=grasp_input,
                        ang_mult=score_model.ang_mult,
                        lin_mult=score_model.lin_mult,
                        n_samples_x_ref=1+(iters%5),
                    )

                with torch.no_grad():
                    _____ = score_model.score_head.warmup(
                        Ts=T.view(-1,7), 
                        key_pcd_multiscale=key_pcd_multiscale,
                        query_pcd=query_pcd,
                        time = time.repeat(len(T))
                    )
                
        score_model.requires_grad_(True)






    