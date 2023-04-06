from typing import Optional, Union, Dict, Tuple, List, Iterable

import torch
import torch.nn as nn
import numpy as np

import theseus as th
from theseus import SO3

from diffusion_edf.transforms import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion, quaternion_apply, quaternion_multiply, se3_exp_map

class SO3_R3():
    def __init__(self, R=None, t=None):
        self.R = SO3()
        if R is not None:
            self.R.update(R)
        self.w = self.R.log_map()
        if t is not None:
            self.t = t

    def log_map(self):
        return torch.cat((self.t, self.w), -1)

    def exp_map(self, x):
        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H
    

class SE3DenoisingDiffusion():

    def __init__(self, field='denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def log_gaussian_on_lie_groups(self, x, context):
        R_p = SO3.exp_map(x[...,3:])
        delta_H = th.compose(th.inverse(context[0]), R_p)
        log = delta_H.log_map()

        dt = x[...,:3] - context[1]

        tlog = torch.cat((dt, log), -1)
        return -0.5 * tlog.pow(2).sum(-1)/(context[2]**2)
    
    def diffuse(self, T_target: torch.Tensor, eps=1e-5, angular_first: bool = False, manual_time: Optional[Union[float, Iterable]] = None):

        T = T_target
        T_in = SO3_R3(R=quaternion_to_matrix(T[..., :4]), t=T[..., 4:])
        tw = T_in.log_map()
        #######################

        ## 1. Compute noisy sample SO(3) + R^3##
        if manual_time:
            if isinstance(manual_time, Iterable):
                random_t = torch.tensor(manual_time, dtype=T_target.dtype, device=T_target.device)
            else:
                random_t = torch.ones_like(tw[...,0], device=tw.device) * manual_time
        else:
            random_t = torch.rand_like(tw[...,0], device=tw.device) * (1. - eps) + eps
        z = torch.randn_like(tw)
        std = self.marginal_prob_std(random_t)
        noise = z * std[..., None]
        noise_t = noise[..., :3]
        noise_rot = SO3.exp_map(noise[...,3:])
        R_p = th.compose(T_in.R, noise_rot)
        t_p = T_in.t + noise_t
        #############################

        ## 2. Compute target score ##
        w_p = R_p.log_map()
        tw_p = torch.cat((t_p, w_p), -1).requires_grad_()
        log_p = self.log_gaussian_on_lie_groups(tw_p, context=[T_in.R, T_in.t, std])
        target_score = torch.autograd.grad(log_p.sum(), tw_p, only_inputs=True)[0]
        target_score = target_score.detach()
        if angular_first:
            target_score = torch.cat([target_score[...,3:], target_score[...,:3]], dim=-1)
        #############################

        ## 3. Get diffusion grad ##
        x_in = tw_p.detach().requires_grad_(True)
        T_in = SO3_R3().exp_map(x_in).to_matrix().detach()
        T_in = torch.cat([standardize_quaternion(matrix_to_quaternion(T_in[..., :3, :3])), T_in[..., :3, -1]], dim=-1)

        time_in = random_t
        
        return target_score, T_in, time_in


def reverse_diffusion(T, score, timestep, angular_first = True):
    if angular_first:
        score = torch.cat([score[..., 3:], score[..., :3]], dim=-1)

    score = score * timestep # Flow
    #score = (score * timestep / 2) + (torch.randn_like(score) * torch.sqrt(timestep)) # Langevin
    reverse_T = se3_exp_map(score)
    reverse_T = torch.cat([standardize_quaternion(matrix_to_quaternion(reverse_T[..., :3, :3])), reverse_T[..., :3, -1]], dim=-1)

    # T = torch.cat([quaternion_multiply(T[..., :4], reverse_T[..., :4]), 
    #                quaternion_apply(T[..., :4], reverse_T[..., 4:]) + T[..., 4:]], dim=-1)
    T = torch.cat([quaternion_multiply(reverse_T[..., :4], T[..., :4]), 
                   quaternion_apply(reverse_T[..., :4], T[..., 4:]) + reverse_T[..., 4:]], dim=-1)
    
    return T, reverse_T