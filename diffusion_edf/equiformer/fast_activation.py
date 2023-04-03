'''
    Functions directly copied from e3nn library.
    
    Speed up some special cases used in GIN and GAT.
'''
from typing import List, Tuple, Optional, Union, Callable
import torch

from e3nn import o3
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.alpha: float = negative_slope
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)


#@compile_mode('script')
class Activation(torch.nn.Module):
    r"""Scalar activation function. 
    Unlike e3nn.nn.Activation, this module directly apply activation when irreps is type-0.

    Odd scalar inputs require activation functions with a defined parity (odd or even).

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------
    Note that 'acts' is a list of nonlinearity (activation).
    >>> a = Activation(irreps_in = "256x0o", acts = [torch.abs])
    >>> a.irreps_out
    256x0e

    Note that 'acts' is a list of nonlinearity (activation).
    >>> a = Activation(irreps_in = "256x0o+16x1e", acts = [torch.nn.SiLU(), None])
    >>> a.irreps_out
    256x0o+16x1e

    'acts' must be 'None' for non-scalar (L>=1) irrep parts.
    >>> a = Activation(irreps_in = "256x0o+16x1e", acts = [torch.nn.SiLU(), torch.nn.SiLU()])
    >>> < ValueError("Activation: cannot apply an activation function to a non-scalar input.") >
    """
    def __init__(self, irreps_in: o3.Irreps, acts: List[Optional[Callable]]):
        """__init__() is Completely Identical to e3nn.nn.Activation.__init__()
        """
        super().__init__()
        irreps_in: o3.Irreps = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts: List[Optional[Callable]] = [normalize2mom(act) if act is not None else None for act in acts] # normalize moment (e3nn functionality)

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError("Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.")
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)

        self.is_acts_none: List[bool] = []
        for i, act in enumerate(acts):
            if act is None:
                acts[i] = torch.nn.Identity()
                self.is_acts_none.append(True)
            else:
                self.is_acts_none.append(False)
        self.acts = torch.nn.ModuleList(acts)
        self.is_acts_none = tuple(self.is_acts_none)
        
        assert len(self.irreps_in) == len(self.acts)

        # If there is only one irrep in o3.Irreps, and the irrep is a scalar, then just apply the only activation to it.
        # For example, "8x0e" is the case.
        # On the other hand, "8x0e+7x0e", "8x1e", "8x0e+7x1e" is not the case.
        if len(self.acts) == 1 and self.acts[0] is not None: # activation for non-scalar irrep cannot be 'None', thus the only irrep must be scalar (L=0).
            self.simple: bool = True
        else:
            self.simple: bool = False

    #def __repr__(self):
    #    acts = "".join(["x" if a is not None else " " for a in self.acts])
    #    return f"{self.__class__.__name__} [{self.acts}] ({self.irreps_in} -> {self.irreps_out})"

    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + '{} -> {}, '.format(self.irreps_in, self.irreps_out)
        return output_str
    

    def forward(self, features: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # If there is only one irrep in o3.Irreps, and the irrep is a scalar, then just apply the only activation to it.
        # For example, "8x0e" is the case.
        # On the other hand, "8x0e+7x0e", "8x1e", "8x0e+7x1e" is not the case.
        if self.simple: # activation for non-scalar irrep cannot be 'None', thus the only irrep must be scalar (L=0).
            return self.acts[0](features)
        
        # Otherwise, same behavior as e3nn.nn.Activation.forward()
        output = []
        index = 0
        for (mul, ir), act, is_act_none in zip(self.irreps_in, self.acts, self.is_acts_none):
            if not is_act_none:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * (2*ir[0] + 1)))
            index += mul * (2*ir[0] + 1)

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)
        
    
#@compile_mode('script')
class Gate(torch.nn.Module):
    '''
        1. Use `narrow` to split tensor.
        2. Use `Activation` in this file.
    '''
    def __init__(self, irreps_scalars: o3.Irreps, act_scalars: List[Callable], 
                 irreps_gates: o3.Irreps, act_gates: List[Callable], 
                 irreps_gated: o3.Irreps):
        super().__init__()
        irreps_scalars: o3.Irreps = o3.Irreps(irreps_scalars)
        irreps_gates: o3.Irreps = o3.Irreps(irreps_gates)
        irreps_gated: o3.Irreps = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}")
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates")

        self.irreps_scalars: o3.Irreps = irreps_scalars
        self.irreps_gates: o3.Irreps = irreps_gates
        self.irreps_gated: o3.Irreps = irreps_gated
        self._irreps_in: o3.Irreps = (irreps_scalars + irreps_gates + irreps_gated).simplify()
        
        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars: o3.Irreps = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates: o3.Irreps = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated: o3.Irreps = self.mul.irreps_out

        for (mul, ir), (mul2, ir2) in zip(self.irreps_scalars, irreps_scalars):
            assert mul == mul2 and ir[0] == ir2[0] and ir[1] == ir2[1] 
        for (mul, ir), (mul2, ir2) in zip(self.irreps_gates, irreps_gates):
            assert mul == mul2 and ir[0] == ir2[0] and ir[1] == ir2[1] 
        for (mul, ir), (mul2, ir2) in zip(self.irreps_gated, irreps_gated):
            assert mul == mul2 and ir[0] == ir2[0] and ir[1] == ir2[1] 

        self._irreps_out = irreps_scalars + irreps_gated

        self.scalars_dim: int = self.irreps_scalars.dim
        self.gates_dim: int = self.irreps_gates.dim
        self.gated_dim: int = self.irreps_gated.dim
        self.input_dim: int = self.irreps_in.dim
        assert self.scalars_dim + self.gates_dim + self.gated_dim == self.input_dim
        

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features.shape == (..., scalar + gates + gated)
        assert features.shape[-1] == self.input_dim
        scalars = features.narrow(-1, 0, self.scalars_dim)
        gates = features.narrow(-1, self.scalars_dim, self.gates_dim)
        gated = features.narrow(-1, (self.scalars_dim + self.gates_dim), self.gated_dim)
        
        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars
        return features


    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in


    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out