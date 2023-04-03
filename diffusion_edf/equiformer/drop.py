'''
    Add `extra_repr` into DropPath implemented by timm 
    for displaying more info.
'''

from typing import Union, Optional, List, Tuple

import torch
import torch.nn as nn
from e3nn import o3
import torch.nn.functional as F

from e3nn.util.jit import compile_mode

@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob: float = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#@compile_mode('script')
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        if drop_prob is not None:
            self.drop_prob: float = drop_prob
        else:
            self.drop_prob: float = 0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)
    
#@compile_mode('script')
class GraphDropPath(nn.Module):
    '''
        Consider batch for graph data when dropping paths.
    '''
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        if drop_prob is not None:
            self.drop_prob: float = drop_prob
        else:
            self.drop_prob: float = 0.

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch_size: int = int(batch.max()) + 1
        shape: List[int] = [batch_size] + [1 for _ in range(x.ndim - 1)]  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out
    
    
    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)
    
    
#@compile_mode('script')
class EquivariantDropout(nn.Module):
    def __init__(self, irreps: o3.Irreps, drop_prob: float):
        super().__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        if drop_prob is not None:
            self.drop_prob: float = drop_prob
        else:
            self.drop_prob: float = 0.
        self.drop = torch.nn.Dropout(self.drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps, 
            o3.Irreps('{}x0e'.format(self.num_irreps)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out
    
#@compile_mode('script')
class EquivariantScalarsDropout(nn.Module):
    def __init__(self, irreps: o3.Irreps, drop_prob: float):
        super(EquivariantScalarsDropout, self).__init__()
        self.irreps: o3.Irreps = irreps
        if drop_prob is not None:
            self.drop_prob: float = drop_prob
        else:
            self.drop_prob: float = 0.
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        out = []
        start_idx = 0
        for mul, ir in self.irreps:
            # temp = x.narrow(-1, start_idx, mul * ir.dim)
            # start_idx += mul * ir.dim
            # if ir.is_scalar():
            #     temp = F.dropout(temp, p=self.drop_prob, training=self.training)
            # out.append(temp)
            temp = x.narrow(-1, start_idx, mul * (2*ir[0] + 1))
            start_idx += mul * (2*ir[0] + 1)
            if ir[0] == 0 and ir[1] == 1:
                temp = F.dropout(temp, p=self.drop_prob, training=self.training)
            out.append(temp)
        out = torch.cat(out, dim=-1)
        return out
    
    
    def extra_repr(self):
        return 'irreps={}, drop_prob={}'.format(self.irreps, self.drop_prob)
    