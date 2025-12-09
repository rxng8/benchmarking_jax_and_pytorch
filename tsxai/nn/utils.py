"""
File: utils.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Heavily adapted some modules from Danijar Hafner's implementation
"""

import collections
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import checkify
# from tensorflow_probability.substrates import jax as tfp
import functools

from . import ninjax as nj

# tfd = tfp.distributions
# tfb = tfp.bijectors
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
f32 = jnp.float32
i32 = jnp.int32

COMPUTE_DTYPE = jnp.bfloat16
PARAM_DTYPE = jnp.float32
ENABLE_CHECKS = False
LAYER_CALLBACK = lambda tensor, name: tensor


@functools.partial(jax.custom_vjp, nondiff_argnums=[1, 2])
def ensure_dtypes(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert x.dtype == fwd, (x.dtype, fwd)
  return x
def ensure_dtypes_fwd(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  return ensure_dtypes(x, fwd, bwd), ()
def ensure_dtypes_bwd(fwd, bwd, cache, dx):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert dx.dtype == bwd, (dx.dtype, bwd)
  return (dx,)
ensure_dtypes.defvjp(ensure_dtypes_fwd, ensure_dtypes_bwd)


def cast_to_compute(values):
  return treemap(
    lambda x: x if x.dtype == COMPUTE_DTYPE else x.astype(COMPUTE_DTYPE),
    values)

def cast_to_compute(xs, force=True):
  if force:
    should = lambda x: True
  else:
    should = lambda x: jnp.issubdtype(x.dtype, jnp.floating)
  return jax.tree.map(lambda x: COMPUTE_DTYPE(x) if should(x) else x, xs)


def cast_to_param(xs, force=True):
  if force:
    should = lambda x: True
  else:
    should = lambda x: jnp.issubdtype(x.dtype, jnp.floating)
  return jax.tree.map(lambda x: PARAM_DTYPE(x) if should(x) else x, xs)


def get_compute_dtype():
  return COMPUTE_DTYPE


def get_param_dtype():
  return PARAM_DTYPE

