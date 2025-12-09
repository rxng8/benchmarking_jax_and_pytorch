"""
File: functional.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Include some functional operations
"""

from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
import math
# from tensorflow_probability.substrates import jax as tfp
# tfd = tfp.distributions
from . import ninjax as nj
from .utils import cast_to_compute, sg
f32 = jnp.float32


def gelu_tanh(x):
  # Constants used in the approximation
  sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
  coeff = 0.044715
  # GELU approximation formula
  return 0.5 * x * (1 + jnp.tanh(sqrt_2_over_pi * (x + coeff * jnp.power(x, 3))))



def dropout(x, prob, training):
  if not prob or not training:
    return x
  keep = jax.random.bernoulli(nj.seed(), 1.0 - prob, x.shape)
  return x * keep / (1.0 - prob)

def rms(xs):
  """Compute root mean square for the whole tree

  Args:
      xs (_type_): _description_

  Returns:
      _type_: _description_
  """
  xs = jax.tree.leaves(xs)
  count = sum(x.size for x in xs)
  sumsq = jnp.stack([f32(jnp.square(x).sum()) for x in xs]).sum()
  return jnp.sqrt(sumsq / f32(count))


def normalize(x, p=2, axis=-1, eps=1e-12):
  norm = jnp.linalg.norm(x, ord=p, axis=axis, keepdims=True)
  return x / (norm + eps)


def where(condition, xs, ys):
  """

  Args:
      condition (Tree): any tree with shape (*some_shape,)
      xs (Tree): any tree with shape (*some_shape, *dim)
      ys (Tree): any tree with shape (*some_shape, *dim)

  Returns:
      Tree: same shape as xs and ys. Resulted masked value with mask broadcasted to the shape of xs and ys
  """
  assert condition.dtype == bool, condition.dtype
  def fn(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    expanded = jnp.expand_dims(condition, list(range(condition.ndim, x.ndim)))
    return jnp.where(expanded, x, y)
  return jax.tree.map(fn, xs, ys)


def mask(xs, mask):
  """Resulted masked value with mask broadcasted to the shape of xs
    negative mask values will become zeros.

  Args:
      xs (Tree): any tree with shape (*some_shape, *dim)
      mask (jax.Array): (*some_shape,)

  Returns:
      Tree: same shape as xs. Resulted masked value with mask broadcasted to the shape of xs
        negative mask values will become zeros.
  """
  return where(mask, xs, jax.tree.map(jnp.zeros_like, xs))
