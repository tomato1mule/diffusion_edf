from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3


from diffusion_edf import transforms
from diffusion_edf.equiformer.graph_attention_transformer import SeparableFCTP
from diffusion_edf.multiscale_tensor_field import MultiscaleTensorField
from diffusion_edf.gnn_data import FeaturedPoints, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffusion_edf.radial_func import SinusoidalPositionEmbeddings


class EbmScoreModelHead(torch.nn.Module):
    jittable: bool = False
    max_time: float
    time_emb_mlp: List[int]
    key_edf_dim: int
    query_edf_dim: int
    n_irreps_prescore: int
    lin_mult: float
    ang_mult: float
    edge_time_encoding: bool
    query_time_encoding: bool
    n_scales: int
    energy_rescale_factor: torch.jit.Final[float]
    q_indices: torch.Tensor
    q_factor: torch.Tensor

    @beartype
    def __init__(self, 
                 max_time: float,
                 time_emb_mlp: List[int],
                 key_tensor_field_kwargs: Dict,
                 irreps_query_edf: Union[str, o3.Irreps],
                 lin_mult: float,
                 ang_mult: float,
                 time_enc_n: float = 10000., 
                 edge_time_encoding: bool = False,
                 query_time_encoding: bool = True):
        super().__init__()
        self.lin_mult = lin_mult
        self.ang_mult = ang_mult
        if 'n_scales' in key_tensor_field_kwargs.keys():
            self.n_scales = key_tensor_field_kwargs['n_scales']
        else:
            self.n_scales = len(key_tensor_field_kwargs['r_cluster_multiscale'])

        ########### Time Encoder #############
        self.time_emb_mlp = time_emb_mlp
        self.irreps_time_emb = o3.Irreps(f"{self.time_emb_mlp[-1]}x0e")
        self.time_enc = SinusoidalPositionEmbeddings(dim=self.time_emb_mlp[0], max_val=max_time, n=time_enc_n)
        time_mlps_multiscale = torch.nn.ModuleList()
        for n in range(self.n_scales):
            time_mlp = []
            for i in range(1,len(time_emb_mlp)):
                time_mlp.append(torch.nn.Linear(self.time_emb_mlp[i-1], self.time_emb_mlp[i]))
                if i != len(time_emb_mlp) -1:
                    time_mlp.append(torch.nn.SiLU(inplace=True))
            time_mlp = torch.nn.Sequential(*time_mlp)
            time_mlps_multiscale.append(time_mlp)
        self.time_mlps_multiscale = time_mlps_multiscale
        if query_time_encoding:
            time_mlp = []
            for i in range(1,len(time_emb_mlp)):
                time_mlp.append(torch.nn.Linear(self.time_emb_mlp[i-1], self.time_emb_mlp[i]))
                if i != len(time_emb_mlp) -1:
                    time_mlp.append(torch.nn.SiLU(inplace=True))
            self.query_time_mlp = torch.nn.Sequential(*time_mlp)
        else:
            self.query_time_mlp = None
        self.time_emb_dim = time_emb_mlp[-1]

        self.edge_time_encoding = edge_time_encoding
        self.query_time_encoding = query_time_encoding
        if not self.edge_time_encoding and not self.query_time_encoding:
            # raise NotImplementedError("No time encoding! Are you sure?")
            pass

        ################# Key field ########################
        if self.query_time_encoding:
            assert 'irreps_query' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['irreps_query'] = str(self.irreps_time_emb)
        else:
            assert 'irreps_query' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['irreps_query'] = None

        if self.edge_time_encoding:
            assert 'edge_context_emb_dim' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['edge_context_emb_dim'] = self.time_emb_mlp[-1]
        else:
            assert 'edge_context_emb_dim' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['edge_context_emb_dim'] = None

        self.key_tensor_field = MultiscaleTensorField(**key_tensor_field_kwargs)
        if self.query_time_encoding:
            assert self.key_tensor_field.use_dst_feature is True

        self.irreps_key_edf = self.key_tensor_field.irreps_output
        self.key_edf_dim = self.irreps_key_edf.dim

        if self.edge_time_encoding:
            assert self.time_emb_dim == self.key_tensor_field.context_emb_dim

        ##################### Query Transform ########################
        self.irreps_query_edf = o3.Irreps(irreps_query_edf)
        self.query_edf_dim = self.irreps_query_edf.dim
        self.query_transform = TransformPcd(irreps = self.irreps_query_edf)
        
        
        ##################### EBM ########################
        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long), persistent=False)
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]), persistent=False)
        self.energy_rescale_factor = 1./float(self.key_edf_dim) # math.sqrt(self.key_edf_dim)
        # self.energy_rescale_factor = torch.nn.Parameter(torch.tensor(1./float(self.key_edf_dim))) 
        self.inference_mode: bool = False
        
    def compute_energy(self, Ts: torch.Tensor,
                       key_pcd_multiscale: List[FeaturedPoints],
                       query_pcd: FeaturedPoints,
                       time: torch.Tensor) -> torch.Tensor:
        # !!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!
        # Batched forward is not yet implemented
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        assert Ts.ndim == 2 and Ts.shape[-1] == 7, f"{Ts.shape}" # Ts: (nT, 4+3: quaternion + position) 
        assert time.ndim == 1 and len(time) == len(Ts), f"{time.shape}" # time: (nT,)
        assert query_pcd.f.ndim == 2 and query_pcd.f.shape[-1] == self.query_edf_dim, f"{query_pcd.f.shape}" # query_pcd: (nQ, 3), (nQ, F), (nQ,), (nQ)

        nT = len(Ts)
        nQ = len(query_pcd.x)

        query_weight = query_pcd.w     # (nQ,)
        assert isinstance(query_weight, torch.Tensor) # to tell torch.jit.script that it is tensor

        time_embs_multiscale: List[torch.Tensor] = []
        time_enc: torch.Tensor = self.time_enc(time)                       # (nT, time_emb_mlp[0])
        for time_mlp in self.time_mlps_multiscale:
            time_embs_multiscale.append(
                time_mlp(time_enc).unsqueeze(-2).expand(-1, nQ, -1).reshape(nT*nQ, self.time_emb_dim)        # (nT, time_emb_D) -> # (nT*nQ, time_emb_D)
            )        

        ################# TODO: SCRUTINIZE THIS CODE ########################
        query_transformed: FeaturedPoints = self.query_transform(pcd = query_pcd, Ts = Ts)                                     # (nT, nQ, 3), (nT, nQ, F), (nT, nQ,), (nT, nQ,)
        query_features_transformed: torch.Tensor = query_transformed.f.clone()                                                 # (nT, nQ, F)         
        if self.query_time_encoding:
            assert self.query_time_mlp is not None
            query_transformed = set_featured_points_attribute(points=query_transformed, 
                                                                              f=self.query_time_mlp(time_enc).unsqueeze(-2).expand(nT, nQ, self.time_emb_dim),  # (nT, time_emb_D) -> # (nT, nQ, time_emb_D)
                                                                              w=None)    # (nT, nQ, 3), (nT, nQ, time_emb), (nT, nQ,), None
        else:
            query_transformed = set_featured_points_attribute(points=query_transformed, f=torch.empty_like(query_transformed.f), w=None)   # (nT, nQ, 3), (nT, nQ, -), (nT, nQ,), None

        query_transformed = flatten_featured_points(query_transformed)                                         # (nT*nQ, 3), (nT*nQ, time_emb), (nT*nQ,), None
        if self.edge_time_encoding:
            query_transformed = self.key_tensor_field(query_points = query_transformed, 
                                                                      input_points_multiscale = key_pcd_multiscale,
                                                                      context_emb = time_embs_multiscale)                      # (nT*nQ, 3), (nT*nQ, F), (nT*nQ,), (nT*nQ,)
        else:
            # assert self.query_time_encoding is True, f"You need to use at least one (query or edge) time encoding method."
            query_transformed = self.key_tensor_field(query_points = query_transformed, 
                                                                      input_points_multiscale = key_pcd_multiscale,
                                                                      context_emb = None)                                         # (nT*nQ, 3), (nT*nQ, F), (nT*nQ,), (nT*nQ,)                                                         # (nT*nQ, F)
        key_features: torch.Tensor = query_transformed.f
        query_features_transformed = query_features_transformed.view(-1, query_features_transformed.shape[-1])                    # (nT*nQ, F)

        ######################################################################
        energy = (key_features-query_features_transformed).square().sum(dim=-1) * self.energy_rescale_factor # (nT*nQ)
        energy = torch.einsum('q,tq->t', query_weight, energy.view(nT, nQ)) # (N_T,)

        return energy
    
    @torch.jit.export
    def warmup(self, Ts: torch.Tensor,
               key_pcd_multiscale: List[FeaturedPoints],
               query_pcd: FeaturedPoints,
               time: torch.Tensor) -> torch.Tensor:
        return self.compute_energy(Ts=Ts, key_pcd_multiscale=key_pcd_multiscale, query_pcd=query_pcd, time=time)
        
    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.inference_mode = not mode
        if mode:
            self.requires_grad_(True)
        else:
            self.requires_grad_(False)
        

    def forward(self, Ts: torch.Tensor,
                key_pcd_multiscale: List[FeaturedPoints],
                query_pcd: FeaturedPoints,
                time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # !!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!
        # Batched forward is not yet implemented
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        assert Ts.ndim == 2 and Ts.shape[-1] == 7, f"{Ts.shape}" # Ts: (nT, 4+3: quaternion + position) 
        assert time.ndim == 1 and len(time) == len(Ts), f"{time.shape}" # time: (nT,)
        assert query_pcd.f.ndim == 2 and query_pcd.f.shape[-1] == self.query_edf_dim, f"{query_pcd.f.shape}" # query_pcd: (nQ, 3), (nQ, F), (nQ,), (nQ)

        T = Ts.detach().requires_grad_(True)
        logP = -self.compute_energy(
            Ts=T,
            key_pcd_multiscale=key_pcd_multiscale,
            query_pcd=query_pcd,
            time=time
        ) # shape: (nT,)
        
        # logP.sum().backward(inputs=T, create_graph=not self.inference_mode)
        # grad = T.grad
        grad = torch.autograd.grad(outputs=logP.sum(), inputs=T, create_graph=not self.inference_mode)[0] # shape: (nT, 7)
        
        L = T.detach()[...,self.q_indices] * self.q_factor
        ang_vel = torch.einsum('...ia,...i', L, grad[...,:4]) * self.ang_mult
        lin_vel = transforms.quaternion_apply(transforms.quaternion_invert(T[...,:4].detach()), grad[...,4:]) * self.lin_mult
        
        if self.inference_mode:
            ang_vel, lin_vel = ang_vel.detach(), lin_vel.detach()

        return ang_vel, lin_vel
    
    @torch.jit.ignore
    def _get_fake_input(self):
        device = next(iter(self.parameters())).device
        
        from diffusion_edf.transforms import random_quaternions
        nT = 5
        nP = 100
        nQ = 10
        Ts = torch.cat([random_quaternions(nT, device=device), torch.randn(nT, 3, device=device)], dim=-1)
        time= torch.rand(nT, device=device)
        
        
        key_pcd_multiscale = [
            FeaturedPoints(
                x=torch.randn(nP,3, device=device, ),
                f=o3.Irreps(self.irreps_key_edf).randn(nP,-1, device=device, ),
                b=torch.zeros(nP, device=device, dtype=torch.long)    
            ) for _ in range(self.n_scales)
        ]
        query_pcd= FeaturedPoints(
                x=torch.randn(nQ,3, device=device, ),
                f=o3.Irreps(self.irreps_key_edf).randn(nQ,-1, device=device, ),
                b=torch.zeros(nQ, device=device, dtype=torch.long),
                w=torch.ones(nQ, device=device, )    
            )
    
        return Ts, key_pcd_multiscale, query_pcd, time
        
        
        
        
        
        