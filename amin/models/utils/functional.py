"""
File: functional.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Include some functional operations
"""

from typing import List, Tuple, Optional
import numpy as np
import math
import torch
from torch.nn import functional as F

from ..utils import tree

def get_act(name: str):
  if callable(name):
    return name
  elif name == 'none' or name == 'identity' or name == None:
    return lambda x: x
  elif name == 'relu':
    return F.relu
  elif name == 'gelu_tanh':
    return gelu_tanh
  elif name == 'gelu_quick':
    return lambda x: x * torch.sigmoid(1.702 * x)
  elif name == 'relu2':
    return lambda x: torch.square(F.relu(x))
  elif name == 'swiglu':
    def fn(x):
      x, y = torch.split(x, 2, -1)
      return F.silu(x) * y
    return fn
  elif name == 'mish':
    return lambda x: x * torch.tanh(F.softplus(x))
  elif hasattr(F, name):
    return getattr(F, name)
  else:
    raise NotImplementedError(name)

def get_act_module_class(name: str) -> callable:
  if callable(name):
    return name
  elif name == 'none' or name == 'identity' or name == None:
    return torch.nn.Identity
  elif name == 'relu':
    return torch.nn.ReLU
  elif name == 'silu':
    return torch.nn.SiLU
  elif name == 'gelu':
    return torch.nn.GELU
  elif name == 'leaky_relu':
    return torch.nn.LeakyReLU
  elif hasattr(torch.nn, name):
    return getattr(torch.nn, name)
  else:
    raise NotImplementedError(name)


def gelu_tanh(x):
  # Constants used in the approximation
  sqrt_2_over_pi = (2.0 / torch.pi)**0.2
  coeff = 0.044715
  # GELU approximation formula
  return 0.5 * x * (1 + torch.tanh(sqrt_2_over_pi * (x + coeff * torch.pow(x, 3))))


def where(condition: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor):
  """

  Args:
      condition (Tree): any tree with shape (*some_shape,)
      xs (Tree): any tree with shape (*some_shape, *dim)
      ys (Tree): any tree with shape (*some_shape, *dim)

  Returns:
      Tree: same shape as xs and ys. Resulted masked value with mask broadcasted to the shape of xs and ys
  """
  assert condition.dtype == torch.bool, condition.dtype
  def fn(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape, (x.shape, y.shape)
    expanded = condition.view(*condition.shape, *([1] * (x.ndim - condition.ndim)))
    return torch.where(expanded, x, y)
  return tree.map(fn, xs, ys)


def mask(xs, mask):
  """Resulted masked value with mask broadcasted to the shape of xs
    negative mask values will become zeros.

  Args:
      xs (Tree): any tree with shape (*some_shape, *dim)
      mask (torch.Tensor): (*some_shape,)

  Returns:
      Tree: same shape as xs. Resulted masked value with mask broadcasted to the shape of xs
        negative mask values will become zeros.
  """
  return where(mask, xs, tree.map(torch.zeros_like, xs))

