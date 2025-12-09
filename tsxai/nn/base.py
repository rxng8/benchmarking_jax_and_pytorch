"""
File: base.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Adapted from Danijar Hafner's code
"""

from typing import Callable, Tuple, List, Dict
import math
import jax
import jax.numpy as jnp
import numpy as np
# from tensorflow_probability.substrates import jax as tfp
import jax.ad_checkpoint as adc

from . import utils as jaxutils
from . import utils
from . import ninjax as nj
from . import functional as F

f32 = jnp.float32
i32 = jnp.int32
# tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
castpd = jaxutils.cast_to_param


def get_act(name):
  if callable(name):
    return name
  elif name == 'none' or name == 'identity' or name == None:
    return lambda x: x
  elif name == 'relu':
    # JAX's relu does not have gradient at 0: https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L54-L88
    # f'(x) can have gradient of 1 at 0. https://stackoverflow.com/a/76396054/14861798
    @jax.custom_jvp
    @jax.jit
    def custom_relu(x):
      return jnp.maximum(x, 0)
    custom_relu.defjvps(lambda g, ans, x: jax.lax.select(x >= 0, g, jax.lax.full_like(g, 0)))
    return custom_relu
  elif name == 'gelu_tanh':
    return F.gelu_tanh
  elif name == 'gelu_quick':
    return lambda x: x * jax.nn.sigmoid(1.702 * x)
  elif name == 'relu2':
    return lambda x: jnp.square(jax.nn.relu(x))
  elif name == 'swiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      return jax.nn.silu(x) * y
    return fn
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)


def get_initializer(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return Initializer(dist, fan, 1.0)


class Initializer:

  def __init__(self, dist='trunc_normal', fan='in', scale=1.0):
    self.dist = dist
    self.fan = fan
    self.scale = scale

  def __call__(self, shape, dtype=jnp.float32, fshape=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    assert all(isinstance(x, int) for x in shape), (
      shape, [type(x) for x in shape])
    assert all(x > 0 for x in shape), shape
    fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
    fan = {
      'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
    }[self.fan]
    if self.dist == 'zeros':
      x = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      x = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      x = jax.random.normal(nj.seed(), shape)
      x *= np.sqrt(1 / fan)
    elif self.dist == 'trunc_normal':
      x = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      x *= 1.1368 * np.sqrt(1 / fan)
    elif self.dist == 'normed':
      x = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      x *= (1 / jnp.linalg.norm(x.reshape((-1, shape[-1])), 2, 0))
    else:
      raise NotImplementedError(self.dist)
    x *= self.scale
    x = x.astype(dtype)
    return x

  def __repr__(self):
    return f'Initializer({self.dist}, {self.fan}, {self.scale})'

  def __eq__(self, other):
    attributes = ('dist', 'fan', 'scale')
    return all(getattr(self, k) == getattr(other, k) for k in attributes)

  @staticmethod
  def compute_fans(shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = math.prod(shape[:-2])
      return (shape[-2] * space, shape[-1] * space)


def get_initializer_pure(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return InitializerPure(dist, fan, 1.0)



class Linear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units: Tuple[int] | int):
    self.units = (units,) if isinstance(units, int) else tuple(units)

  def macs(self, x):
    # x: (..., input_features)
    insize = x.shape[-1]
    size = math.prod(self.units)
    return int(math.prod(x.shape[:-1]) * insize * size)

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    size = math.prod(self.units)
    shape = (x.shape[-1], size)
    x = x @ self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), size).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], *self.units))
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale



class Norm(nj.Module):

  axis: tuple = (-1,)
  eps: float = 1e-4
  scale: bool = True
  shift: bool = True

  def __init__(self, impl):
    if '1em' in impl:
      impl, exp = impl.split('1em')
      self._fields['eps'] = 10 ** -int(exp)
    self.impl = impl

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    dtype = x.dtype
    x = f32(x)
    axis = [a % x.ndim for a in self.axis]
    shape = [x.shape[i] if i in axis else 1 for i in range(min(axis), x.ndim)]
    if self.impl == 'none':
      pass
    elif self.impl == 'rms':
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      scale = self._scale(shape, x.dtype)
      x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
    elif self.impl == 'layer':
      mean = x.mean(axis, keepdims=True)
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = adc.checkpoint_name(var, 'small')
      scale = self._scale(shape, x.dtype)
      shift = self._shift(shape, x.dtype)
      x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    else:
      raise NotImplementedError(self.impl)
    x = x.astype(dtype)
    return x

  def macs(self, x):
    """Estimate MACs for this normalization on input x.

    This is a heuristic estimate (ops per element) because exact cost depends
    on implementation (reductions, scaling, optional parameters).
    """
    # Total elements in tensor
    try:
      total_elems = int(math.prod(x.shape))
    except Exception:
      # If shape is not available, return 0
      return 0

    # Heuristic ops per element by impl
    if self.impl == 'none':
      ops_per_elem = 0
    elif self.impl == 'rms':
      # square, mean (reduce), add eps, rsqrt, multiply, maybe scale
      ops_per_elem = 5
    elif self.impl == 'layer':
      # mean, mean2, var (sub/square), rsqrt, multiply, shift -> more ops
      ops_per_elem = 10
    else:
      # default conservative estimate
      ops_per_elem = 6

    return int(total_elems * ops_per_elem)

  def _scale(self, shape, dtype):
    if not self.scale:
      return jnp.ones(shape, dtype)
    return self.value('scale', jnp.ones, shape, f32).astype(dtype)

  def _shift(self, shape, dtype):
    if not self.shift:
      return jnp.zeros(shape, dtype)
    return self.value('shift', jnp.zeros, shape, f32).astype(dtype)


class MLP(nj.Module):

  act: str = 'silu'
  norm: str = 'rms'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, layers: int = 5, units: int = 1024):
    self.layers = layers
    self.units = units
    self.kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    self.outkw = dict(outscale=self.outscale, **self.kw)

  def __call__(self, x):
    shape = x.shape[:-1]
    x = x.astype(utils.COMPUTE_DTYPE)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self.layers):
      kw = self.kw if i < self.layers - 1 else self.outkw
      x = self.sub(f'linear{i}', Linear, self.units, **kw)(x)
      x = self.sub(f'norm{i}', Norm, self.norm)(x)
      x = get_act(self.act)(x)
    x = x.reshape((*shape, x.shape[-1]))
    return x

  def macs(self, x):
    # x may have leading batch dims; we reshape similarly to __call__
    try:
      bshape = x.shape[:-1]
      flat_n = int(math.prod(bshape)) if bshape else 1
      in_ch = int(x.shape[-1])
    except Exception:
      return 0

    total = 0
    for i in range(self.layers):
      kw = self.kw if i < self.layers - 1 else self.outkw
      out_units = int(self.units)
      # linear macs
      try:
        lin = Linear(out_units)
        total += int(lin.macs(jnp.zeros((flat_n, in_ch))))
      except Exception:
        total += int(flat_n * in_ch * out_units)
      # norm macs
      try:
        total += int(Norm(self.norm).macs(jnp.zeros((flat_n, out_units))))
      except Exception:
        total += int(flat_n * out_units * 6)
      # activation (~4 ops per element)
      total += int(flat_n * out_units * 4)
      in_ch = out_units

    return int(total)


class GRU(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int):
    self.units = units

  def initial(self, batch_size):
    return jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE)

  def macs(self, carry, inputs, resets):
    """Compute MACs for GRU operation.
    
    Args:
        carry (jax.Array): (B, U)
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)
    
    Returns:
        int: Total number of multiply-accumulate operations
    """
    try:
      B, T, I = inputs.shape
      U = self.units
      
      # Per timestep operations:
      # 1. Concatenation: no MACs
      # 2. Normalization (RMS/Layer): ~6 ops per element for (B, U+I)
      norm_ops = B * (U + I) * 6
      
      # 3. Linear layer: (B, U+I) @ (U+I, 3*U) -> (B, 3*U)
      linear_macs = B * (U + I) * (3 * U)
      
      # 4. Element-wise operations per unit:
      #    - sigmoid(res): ~4 ops per element (U elements)
      #    - multiply (res * cand): 1 MAC per element (U elements)
      #    - tanh(result): ~4 ops per element (U elements)
      #    - sigmoid(update): ~4 ops per element (U elements)
      #    - update * cand: 1 MAC per element (U elements)
      #    - (1 - update): 1 op per element (U elements)
      #    - (1 - update) * carry: 1 MAC per element (U elements)
      #    - final addition: 1 op per element (U elements)
      # Total: ~17 ops per unit
      elementwise_ops = B * U * 17
      
      # Total per timestep
      per_step = norm_ops + linear_macs + elementwise_ops
      
      # Multiply by number of timesteps
      total_macs = int(T * per_step)
      
      return total_macs
    except Exception:
      # Fallback if shape information is not available
      return 0

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): (B, U)
        inputs (jax.Array): (B, I) or (B, T, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert carry.dtype == utils.COMPUTE_DTYPE, carry.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    # print(f"[GRU] carry: {carry.shape}, inputs: {inputs.shape}, resets: {resets.shape}")
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = F.mask(carry, ~reset)
    # print(f"[GRU.step] carry: {carry.shape}, inp: {inp.shape}")
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('norm', Norm, self.norm)(x)
    x = self.sub('linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output

