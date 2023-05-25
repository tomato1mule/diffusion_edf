from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3


from diffusion_edf import transforms
from diffusion_edf.equiformer.graph_attention_transformer import SeparableFCTP
from diffusion_edf.feature_extractor import UnetFeatureExtractor
from diffusion_edf.tensor_field import TensorField
from diffusion_edf.gnn_data import FeaturedPoints, GraphEdge, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffusion_edf.radial_func import SinusoidalPositionEmbeddings
from diffusion_edf.wigner import TransformFeatureQuaternion


class ScoreModelHead(torch.nn.Module):
    max_time: float
    time_emb_mlp: List[int]
    key_edf_dim: int
    query_edf_dim: int
    n_irreps_prescore: int
    lin_mult: float
    ang_mult: float
    use_time_emb_for_edge_encoding: bool

    @beartype
    def __init__(self, 
                 max_time: float,
                 time_emb_mlp: List[int],
                 key_tensor_field_kwargs: Dict,
                 irreps_query_edf: Union[str, o3.Irreps],
                 lin_mult: float,
                 ang_mult: float = math.sqrt(2),
                 use_time_emb_for_edge_encoding: bool = False):
        super().__init__()
        self.lin_mult = lin_mult
        self.ang_mult = ang_mult

        ########### Time Encoder #############
        self.time_emb_mlp = time_emb_mlp
        self.irreps_time_emb = o3.Irreps(f"{self.time_emb_mlp[-1]}x0e")
        time_enc = [SinusoidalPositionEmbeddings(dim=self.time_emb_mlp[0], max_val=max_time, n=10000.)]
        for i in range(1,len(time_emb_mlp)):
            time_enc.append(torch.nn.SiLU(inplace=True))
            time_enc.append(torch.nn.Linear(self.time_emb_mlp[i-1], self.time_emb_mlp[i]))
        time_enc.append(torch.nn.SiLU(inplace=True)) # because there is another mlp in the key_tensor_field
        self.time_enc = torch.nn.Sequential(*time_enc)
        self.use_time_emb_for_edge_encoding = use_time_emb_for_edge_encoding

        ################# Key field ########################
        if 'irreps_query' not in key_tensor_field_kwargs.keys():
            key_tensor_field_kwargs['irreps_query'] = str(self.irreps_time_emb)
        assert o3.Irreps(key_tensor_field_kwargs['irreps_query']) == self.irreps_time_emb, f"{key_tensor_field_kwargs['irreps_query']}"

        self.key_tensor_field = TensorField(**key_tensor_field_kwargs)
        self.irreps_key_edf = self.key_tensor_field.irreps_output
        self.key_edf_dim = self.irreps_key_edf.dim

        ##################### Query Transform ########################
        self.irreps_query_edf = o3.Irreps(irreps_query_edf)
        self.query_edf_dim = self.irreps_query_edf.dim
        self.query_transform = TransformPcd(irreps = self.irreps_query_edf)

        ##################### Tensor product for lin/ang score ###################
        self.n_irreps_prescore = 0
        for mul, (l, p) in self.irreps_query_edf:
            if l == 1:
                assert p == 1
                self.n_irreps_prescore += mul
        for mul, (l, p) in self.irreps_key_edf:
            if l == 1:
                assert p == 1
                self.n_irreps_prescore += mul
        self.n_irreps_prescore = self.n_irreps_prescore // 2

        self.irreps_prescore = o3.Irreps(f"{self.n_irreps_prescore}x1e")
        self.lin_vel_tp = SeparableFCTP(irreps_node_input = self.irreps_key_edf,
                                        irreps_edge_attr = self.irreps_query_edf, 
                                        irreps_node_output = o3.Irreps("1x0e") + self.irreps_prescore,  # Append 1x0e to avoid torch jit error. TODO: Remove this
                                        fc_neurons = None, 
                                        use_activation = True, 
                                        #norm_layer = 'layer', 
                                        norm_layer = None,
                                        internal_weights = True)
        #self.lin_vel_proj = LinearRS(irreps_in = self.irreps_prescore, irreps_out = o3.Irreps("1x1e"), bias=False, rescale=False).to(device)
        self.ang_vel_tp = SeparableFCTP(irreps_node_input = self.irreps_key_edf,
                                        irreps_edge_attr = self.irreps_query_edf, 
                                        irreps_node_output = o3.Irreps("1x0e") + self.irreps_prescore,  # Append 1x0e to avoid torch jit error. TODO: Remove this
                                        fc_neurons = None, 
                                        use_activation = True, 
                                        #norm_layer = 'layer', 
                                        norm_layer = None,
                                        internal_weights = True)
        #self.ang_vel_proj = LinearRS(irreps_in = self.irreps_prescore, irreps_out = o3.Irreps("1x1e"), bias=False, rescale=False).to(device)

    @torch.jit.ignore()
    def to(self, *args, **kwargs):
        for module in self.children():
            if isinstance(module, torch.nn.Module):
                module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, Ts: torch.Tensor,
                key_pcd: FeaturedPoints,
                query_pcd: FeaturedPoints,
                time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # !!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!
        # Batched forward is not yet implemented
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        assert Ts.ndim == 2 and Ts.shape[-1] == 7, f"{Ts.shape}" # Ts: (nT, 4+3: quaternion + position) 
        assert time.ndim == 1 and len(time) == len(Ts), f"{time.shape}" # time: (nT,)
        assert query_pcd.f.ndim == 2 and query_pcd.f.shape[-1] == self.query_edf_dim, f"{query_pcd.f.shape}" # query_pcd: (nQ, 3), (nQ, F), (nQ,), (nQ)

        nT = len(Ts)
        nQ = len(query_pcd.x)

        query_weight: torch.Tensor = query_pcd.w     # (nQ,)
        time_emb: torch.Tensor = self.time_enc(time) # (nT, time_emb_D)
        time_emb = time_emb.unsqueeze(-2).expand(-1, nQ, -1) # (nT, nQ, time_emb_D)
        

        query_transformed: FeaturedPoints = self.query_transform(pcd = query_pcd, Ts = Ts)                                     # (nT, nQ, 3), (nT, nQ, F), (nT, nQ,), (nT, nQ,)
        query_transformed: FeaturedPoints = set_featured_points_attribute(points=query_transformed, f=time_emb, w=None)        # (nT, nQ, 3), (nT, nQ, time_emb), (nT, nQ,), None
        query_transformed: FeaturedPoints = flatten_featured_points(query_transformed)                                         # (nT*nQ, 3), (nT*nQ, time_emb), (nT*nQ,), None
        if self.use_time_emb_for_edge_encoding:
            query_transformed: FeaturedPoints = self.key_tensor_field(query_points = query_transformed, 
                                                                      input_points = key_pcd,
                                                                      time_emb = query_transformed.f)                          # (nT*nQ, 3), (nT*nQ, F), (nT*nQ,), (nT*nQ,)
        else:
            query_transformed: FeaturedPoints = self.key_tensor_field(query_points = query_transformed, 
                                                                      input_points = key_pcd,
                                                                      time_emb = None)                                         # (nT*nQ, 3), (nT*nQ, F), (nT*nQ,), (nT*nQ,)
        query_features_transformed: torch.Tensor = query_transformed.f                                                         # (nT*nQ, F)
        key_features: torch.Tensor = query_transformed.f                                                                       # (nT*nQ, F)

        lin_vel: torch.Tensor = self.lin_vel_tp(query_features_transformed, key_features,   # (nT*nQ, 1+F_prescore)
                                                edge_scalars = None, batch=None,)           # batch does nothing unless you use batchnorm
        ang_spin: torch.Tensor = self.ang_vel_tp(query_features_transformed, key_features,  # (nT*nQ, 1+F_prescore)
                                                 edge_scalars = None, batch=None)           # batch does nothing unless you use batchnorm
        lin_vel, ang_spin = lin_vel[..., 1:], ang_spin[..., 1:] # Discard the placeholder 1x0e feature to avoid torch jit error. TODO: Remove this

        lin_vel = lin_vel.view(nT, nQ, self.n_irreps_prescore, 3).mean(dim=-2)    # (N_T, N_Q, 3), Project multiple nx1e -> 1x1e 
        ang_spin = ang_spin.view(nT, nQ, self.n_irreps_prescore, 3).mean(dim=-2)  # (N_T, N_Q, 3), Project multiple nx1e -> 1x1e 

        q = Ts[..., :4]
        qinv: torch.Tensor = transforms.quaternion_invert(q.unsqueeze(-2)) # (N_T, 1, 4)
        lin_vel = transforms.quaternion_apply(qinv, lin_vel) # (N_T, N_Q, 3)
        ang_spin = transforms.quaternion_apply(qinv, ang_spin) # (N_T, N_Q, 3)
        ang_orbital = torch.cross(query_pcd.x.unsqueeze(0), lin_vel, dim=-1) # (N_T, N_Q, 3)

        lin_vel = torch.einsum('q,tqi->ti', query_weight, lin_vel) / self.lin_mult # (N_T, 3)
        ang_vel = (torch.einsum('q,tqi->ti', query_weight, ang_orbital) / self.lin_mult) \
                +   (torch.einsum('q,tqi->ti', query_weight, ang_spin) / self.ang_mult) # (N_T, 3)

        return ang_vel, lin_vel


        




        




class ScoreModel(torch.nn.Module):
    @beartype
    def __init__(self, 
                 max_time: float,
                 time_emb_mlp: List[int],
                 key_kwargs: Dict,
                 query_kwargs: Dict,
                 lin_mult: float,
                 ang_mult: float = math.sqrt(2),
                 deterministic: bool = False):
        super().__init__()

        key_feature_extractor_kwargs = key_kwargs['feature_extractor_configs']
        key_tensor_field_kwargs = key_kwargs['tensor_field_configs']
        key_tensor_field_kwargs['irreps_input'] = key_feature_extractor_kwargs['irreps_output']
        query_feature_extractor_kwargs = query_kwargs['feature_extractor_configs']
        use_time_emb_for_edge_encoding = key_kwargs['use_time_emb_for_edge_encoding']

        print("ScoreModel: Initializing Score Head")
        self.score_head = ScoreModelHead(max_time=max_time, 
                                         time_emb_mlp=time_emb_mlp,
                                         key_tensor_field_kwargs=key_tensor_field_kwargs,
                                         # irreps_query_edf=self.query_feature_extractor.irreps_output,
                                         irreps_query_edf=query_feature_extractor_kwargs['irreps_output'],
                                         lin_mult=lin_mult,
                                         ang_mult=ang_mult,
                                         use_time_emb_for_edge_encoding = use_time_emb_for_edge_encoding
                                         )

        print("ScoreModel: Initializing Key Feature Extractor")
        self.key_feature_extractor = UnetFeatureExtractor(
            **(key_feature_extractor_kwargs),
            deterministic=deterministic
        )

        print("ScoreModel: Initializing Query Feature Extractor")
        self.query_feature_extractor = UnetFeatureExtractor(
            **(query_feature_extractor_kwargs),
            deterministic=deterministic
        )

        self.lin_mult = self.score_head.lin_mult
        self.ang_mult = self.score_head.ang_mult

        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long), persistent=False)
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]), persistent=False)

    @torch.jit.ignore()
    def to(self, *args, **kwargs):
        for module in self.children():
            if isinstance(module, torch.nn.Module):
                module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, Ts: torch.Tensor, 
                time: torch.Tensor, 
                key_pcd: FeaturedPoints, 
                query_pcd: FeaturedPoints, 
                extract_features: bool = True,
                debug: bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                             Optional[Tuple[FeaturedPoints, FeaturedPoints]]]:
        if extract_features:
            key_pcd: FeaturedPoints = self.key_feature_extractor(key_pcd)
            query_pcd: FeaturedPoints = self.query_feature_extractor(query_pcd)

        score: Tuple[torch.Tensor, torch.Tensor] = self.score_head(Ts = Ts, 
                                                                   key_pcd = key_pcd, 
                                                                   query_pcd = query_pcd,
                                                                   time = time)
        if debug:
            debug_output = (detach_featured_points(key_pcd), detach_featured_points(query_pcd))
        else:
            debug_output = None
                                                                   
        return score, debug_output
