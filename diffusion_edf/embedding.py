#### Deprecated #####

import warnings

import torch
from e3nn.util.jit import compile_mode
from e3nn import o3

from diffusion_edf.equiformer.tensor_product_rescale import LinearRS



#@compile_mode('script')
class NodeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_input: o3.Irreps, irreps_node_emb: o3.Irreps, bias: bool = True, input_mean = torch.tensor([0.5, 0.5, 0.5]), input_std = torch.tensor([0.5, 0.5, 0.5])):
        
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_node_emb = o3.Irreps(irreps_node_emb)
        self.linear = LinearRS(self.irreps_input, 
                                      self.irreps_node_emb, 
                                      bias=bias)
        
        if not isinstance(input_mean, torch.Tensor):
            input_mean = torch.tensor(input_mean)
        if not isinstance(input_std, torch.Tensor):
            input_std = torch.tensor(input_std)
        self.register_buffer("input_mean", input_mean)
        if not (input_std > 1e-4).all():
            warnings.warn(f"Too small input std: {input_std}")
        self.register_buffer("input_std", input_std)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        input_feature = (input_feature - self.input_mean) / self.input_std
        node_embedding: torch.Tensor = self.linear(input_feature)
        
        return node_embedding