import torch
from e3nn.util.jit import compile_mode
from e3nn import o3

from diffusion_edf.equiformer.tensor_product_rescale import LinearRS

#@compile_mode('script')
class NodeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_input: o3.Irreps, irreps_node_emb: o3.Irreps, bias: bool = True):
        
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_node_emb = o3.Irreps(irreps_node_emb)
        self.linear = LinearRS(self.irreps_input, 
                                      self.irreps_node_emb, 
                                      bias=bias)
        
    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        node_embedding: torch.Tensor = self.linear(input_feature)
        
        return node_embedding