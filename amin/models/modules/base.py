# %%

# import sys, pathlib
# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

from typing import List, Union, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F

# from amin.models import utils as U
from .. import utils as U

class Linear(nn.Module):
  """Linear module with weights init"""
  def __init__(self, in_unit: int | None, out_unit: int, bias: bool = True,
      winit: str = 'trunc_normal', binit: str = 'zeros', outscale: float = 1.0) -> None:
    super().__init__()
    self._in_unit = in_unit
    self._out_unit = out_unit
    self.bias = bias
    self.winit = winit
    self.binit = binit

    # Create linear layer with custom initialization if input size is known
    if in_unit is not None:
      self.linear = nn.Linear(self._in_unit, self._out_unit, bias=bias)
      U.initialize_module(self.linear.weight, winit, outscale)
      if bias:
        U.initialize_module(self.linear.bias, binit)
    else:
      self.linear = None  # Will be created on first forward pass

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x (torch.Tensor): (B, I) or (B, T, I)

    Returns:
      torch.Tensor: (B, O) or (B, T, O)
    """
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), x.dtype
    return self.linear(x)


class GRU(nn.Module):
  def __init__(self, in_unit: int, out_unit: int, norm: bool = True,
      bias: bool = True, update_bias: float = -1.0, winit: str = 'trunc_normal',
      binit: str = 'zeros', outscale: float = 1.0) -> None:
    super().__init__()
    self._in_unit = in_unit
    self._out_unit = out_unit
    self._update_bias = update_bias
    self.norm_layer = nn.LayerNorm(self._out_unit + self._in_unit, eps=1e-03) if norm else None

    # Create linear layer with custom initialization
    self.linear = nn.Linear(self._in_unit + self._out_unit, 3 * self._out_unit, bias=bias)
    U.initialize_module(self.linear.weight, winit, outscale)
    if bias:
      U.initialize_module(self.linear.bias, binit)

  def initial(self, batch_size: int, device=None) -> torch.Tensor:
    return torch.zeros(batch_size, self._out_unit, device=device)

  def forward(self, carry: torch.Tensor, inputs: torch.Tensor,
      resets: torch.Tensor, single: bool = True):
    """
    Args:
      carry (torch.Tensor): (B, U) - hidden state
      inputs (torch.Tensor): (B, I) or (B, T, I) - inputs
      resets (torch.Tensor): (B,) or (B, T) - boolean reset mask
      single (bool, optional): If True, process single timestep. Defaults to True.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: (carry, outputs)
        - carry: (B, U) - final hidden state
        - outputs: (B, U) if single=True, else (B, T, U) - all outputs
    """
    assert carry.dtype in (torch.float32, torch.float16, torch.bfloat16), carry.dtype
    assert inputs.dtype in (torch.float32, torch.float16, torch.bfloat16), inputs.dtype
    assert resets.dtype == torch.bool, resets.dtype

    if single:
      carry, output = self.step(carry, inputs, resets)
      return carry, output

    # Scan over time dimension (similar to jax.lax.scan)
    outputs = []
    T = inputs.shape[1]
    for t in range(T):
      carry, output = self.step(carry, inputs[:, t], resets[:, t])
      outputs.append(output)

    outputs = torch.stack(outputs, dim=1)  # (B, T, U)
    return carry, outputs

  def step(self, carry: torch.Tensor, inp: torch.Tensor, reset: torch.Tensor):
    """Single timestep update.

    Args:
        carry (torch.Tensor): (B, U)
        inp (torch.Tensor): (B, I)
        reset (torch.Tensor): (B,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (new_carry, output)
    """
    # Mask carry based on reset (zero out when reset is True)
    carry = U.mask(carry, ~reset)

    # Concatenate carry and input
    x = torch.cat([carry, inp], dim=-1)  # (B, U+I)

    # Apply norm if exists
    if self.norm_layer is not None:
      x = self.norm_layer(x)

    # Linear projection
    x = self.linear(x)  # (B, 3*U)

    # Split into three parts
    res, cand, update = torch.split(x, self._out_unit, dim=-1)

    # GRU equations
    cand = torch.tanh(torch.sigmoid(res) * cand)
    update = torch.sigmoid(update + self._update_bias)
    carry = output = update * cand + (1 - update) * carry

    return carry, output

# model = GRU(4, 8)
# inp = torch.randn(2, 4)
# carry = model.initial(2)
# resets = torch.tensor([0, 1], dtype=torch.bool)
# carry, out = model(carry, inp, resets)

# model = GRU(4, 8)
# inp = torch.randn(2, 5, 4)
# carry = model.initial(2)
# resets = torch.tensor([[0], [1]], dtype=torch.bool).repeat(1, 5)
# carry, out = model(carry, inp, resets, single=False)


class MLP(nn.Module):
  """Multi-layer perceptron with normalization and activation"""
  def __init__(self, in_units: int, layers: int = 5, units: int = 1024, act: str = 'silu',
               norm: str = 'layer', bias: bool = True, winit: str = 'trunc_normal',
               binit: str = 'zeros', outscale: float = 1.0) -> None:
    super().__init__()
    self.in_units = in_units
    self.layers = layers
    self.units = units
    self.act = act
    self.norm = norm
    self.bias = bias
    self.winit = winit
    self.binit = binit

    # Build layers
    self.linears = nn.ModuleList()
    self.norms = nn.ModuleList()

    for i in range(self.layers):
      # Use outscale for the last layer
      is_last = (i == self.layers - 1)
      # Note: outscale would need to be applied to initialization
      self.linears.append(Linear(in_units if i == 0 else units, units, bias=bias,
        winit=winit, binit=binit, outscale=outscale if is_last else 1.0))

      # Add normalization
      if norm == 'layer':
        self.norms.append(nn.LayerNorm(units, eps=1e-03))
      else:
        self.norms.append(nn.Identity())

    # Get activation function
    self.act_fn = U.get_act(act)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x (torch.Tensor): Input tensor of shape (*shape, in_features)

    Returns:
      torch.Tensor: Output tensor of shape (*shape, units)
    """
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    for i in range(self.layers):
      x = self.linears[i](x)
      x = self.norms[i](x)
      x = self.act_fn(x)
    x = x.reshape(*shape, -1)
    return x

# model = MLP(4, layers=3, units=8)
# inp = torch.randn(2, 5, 4)
# out = model(inp)
# out.shape
