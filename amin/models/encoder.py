
"""
Encoder Module for Multi-Modal Observations

This module provides a flexible encoder architecture for processing multi-modal
observations including vector and image inputs.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Optional, Tuple

from .modules.base import Linear, Conv2D, get_norm_layer
from . import utils as U
from .utils import DictConcat

class Encoder(nn.Module):
  """
  A flexible neural network encoder for processing multi-modal observations.

  This encoder can handle both vector and image-based inputs, applying different
  processing strategies based on input type. It supports:
  - Vector inputs processed through multi-layer perceptron (MLP)
  - Image inputs processed through convolutional neural network (CNN)
  - Optional symlog transformation for continuous inputs
  - Configurable network depth, units, normalization, and activation

  Example config:
    # for 192x192 input
    **{depth: 64, mults: [2, 3, 4, 4, 4, 4], layers: 3, units: 1024, act: 'silu', norm: 'rms', winit: 'trunc_normal', symlog: True, outer: False, kernel: 5, strided: False}
  """

  def __init__(
    self,
    obs_space: Dict,
    units: int = 1024,
    norm: str = 'rms',
    act: str = 'gelu',
    depth: int = 64,
    mults: Tuple[int, ...] = (2, 3, 4, 4, 4),
    layers: int = 3,
    kernel: int = 5,
    symlog: bool = True,
    outer: bool = False,
    strided: bool = False,
    winit: str = 'trunc_normal',
    binit: str = 'zeros',
    outscale: float = 1.0,
  ):
    """
    Initialize the encoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining input characteristics
      units (int): Number of units in the MLP layers
      norm (str): Normalization type ('layer', 'rms', 'none')
      act (str): Activation function name
      depth (int): Base depth for convolutional layers
      mults (tuple): Multipliers for increasing depth in CNN layers
      layers (int): Number of MLP layers
      kernel (int): Kernel size for convolutional layers
      symlog (bool): Whether to apply symlog transformation to continuous inputs
      outer (bool): Special handling for first CNN layer
      strided (bool): Whether to use strided convolutions
      winit (str): Weight initialization method
      binit (str): Bias initialization method
      outscale (float): Output scaling factor
    """
    super().__init__()
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(depth * mult for mult in mults)

    self.units = units
    self.norm = norm
    self.act = act
    self.depth = depth
    self.mults = mults
    self.layers = layers
    self.kernel = kernel
    self.symlog = symlog
    self.outer = outer
    self.strided = strided
    self.winit = winit
    self.binit = binit
    self.outscale = outscale

    # Build vector encoder
    if self.veckeys:
      self.mlp_layers = nn.ModuleList()
      self.mlp_norms = nn.ModuleList()
      
      # Calculate input size for vector encoder
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vec_input_size = sum(int(torch.prod(torch.tensor(s.shape))) for s in vspace.values())
      
      for i in range(self.layers):
        in_units = vec_input_size if i == 0 else self.units
        self.mlp_layers.append(
          Linear(in_units, self.units, winit=self.winit, binit=self.binit, outscale=self.outscale)
        )
        self.mlp_norms.append(get_norm_layer(self.norm, self.units))
    
    # Build image encoder
    if self.imgkeys:
      self.cnn_layers = nn.ModuleList()
      self.cnn_norms = nn.ModuleList()
      
      # Calculate input channels
      in_channels = sum(s.shape[-1] for k, s in obs_space.items() if k in self.imgkeys)
      
      for i, d in enumerate(self.depths):
        in_ch = in_channels if i == 0 else self.depths[i - 1]
        
        if self.outer and i == 0:
          padding = self.kernel // 2
          self.cnn_layers.append(
            Conv2D(in_ch, d, self.kernel, stride=1, padding=padding, 
                   winit=self.winit, binit=self.binit, outscale=self.outscale)
          )
        elif self.strided:
          padding = self.kernel // 2
          self.cnn_layers.append(
            Conv2D(in_ch, d, self.kernel, stride=2, padding=padding,
                   winit=self.winit, binit=self.binit, outscale=self.outscale)
          )
        else:
          padding = self.kernel // 2
          self.cnn_layers.append(
            Conv2D(in_ch, d, self.kernel, stride=1, padding=padding,
                   winit=self.winit, binit=self.binit, outscale=self.outscale)
          )
        
        self.cnn_norms.append(get_norm_layer(self.norm, d))
    
    # Get activation function
    self.act_fn = U.get_act(self.act)

  def forward(self, obs: Dict[str, torch.Tensor], reset: torch.Tensor, 
              training: bool = True, single: bool = False) -> torch.Tensor:
    """
    Process multi-modal observations through vector and image encoders.

    Args:
      obs (dict): Input observations dictionary
      reset (torch.Tensor): Reset signal for handling episode boundaries
      training (bool): Training mode flag
      single (bool, optional): Whether processing a single timestep. Defaults to False.

    Returns:
      embed (torch.Tensor): Encoded representation of input observations
    """
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    # Process vector inputs
    if self.veckeys:
      vecs = []
      for k in sorted(self.veckeys):
        x = obs[k]
        if self.symlog:
          x = U.symlog(x)
        # Flatten to (*bshape, -1)
        x = x.reshape(*bshape, -1)
        vecs.append(x)
      x = torch.cat(vecs, dim=-1)
      x = x.reshape(-1, x.shape[-1])

      for i in range(self.layers):
        x = self.mlp_layers[i](x)
        x = self.mlp_norms[i](x)
        x = self.act_fn(x)
      outs.append(x)

    # Process image inputs
    if self.imgkeys:
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == torch.uint8 for x in imgs), "Image inputs must be uint8"
      x = torch.cat(imgs, dim=-1).float() / 255.0 - 0.5
      x = x.reshape(-1, *x.shape[bdims:])
      
      # Convert from (B, H, W, C) to (B, C, H, W) for PyTorch conv
      x = x.permute(0, 3, 1, 2)
      
      for i, depth in enumerate(self.depths):
        x = self.cnn_layers[i](x)
        x = self.cnn_norms[i](x)
        x = self.act_fn(x)
        
        # Apply max pooling for non-strided case
        if not self.strided and not (self.outer and i == 0):
          x = F.max_pool2d(x, kernel_size=2, stride=2)
      
      # Flatten spatial dimensions
      B = x.shape[0]
      x = x.reshape(B, -1)
      outs.append(x)

    # Concatenate all outputs
    x = torch.cat(outs, dim=-1)
    embed = x.reshape(*bshape, *x.shape[1:])
    return embed

