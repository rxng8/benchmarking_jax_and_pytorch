# %%

# Using `bench` conda environment. python=3.12 and jax with cuda 12
import sys, pathlib

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from tsxai import nn
from tsxai import tree
from tsxai import timer

def rand(*shape):
  return np.random.normal(size=shape).astype(np.float32)

def count_params(params: Dict) -> int:
  return sum(tree.map(lambda x: x.size, params).values())


to_jax = lambda x: tree.map(lambda x2: jnp.array(x2), x)


# config
I = 2048 # input dim
O = 4096 # output dim
Z = 4096 # hidden dim
S = 1024 # state dim
B = 32 # batch size
T = 64 # sequence length


class Model(nn.Module):
  units: int
  out_units: int

  def __init__(self):
    self.gru = nn.GRU(units=self.units, name="gru")
    self.mlp = nn.MLP(units=self.out_units, layers=20, name="mlp")

  def initial(self, batch_size: int):
    return self.gru.initial(batch_size)

  def __call__(self, carry, inputs, resets, single=False):
    carry, out = self.gru(carry, inputs, resets, single=single)
    return carry, self.mlp(out)

# jax model
model = Model(units=Z, out_units=O, name="jax_gru")
params = nn.init(model.initial, static_argnames=("batch_size"))({}, batch_size=B, seed=0)
_, carry = nn.pure(model.initial)(params, batch_size=B, seed=0)

inputs = nn.cast_to_compute(jnp.asarray(rand(B, T, I)))
resets = jnp.zeros((B, T), dtype=bool)
params = nn.init(model, static_argnames=("single"))(params, carry, inputs, resets, single=False, seed=1)
forward = jax.jit(nn.pure(model), static_argnames=("single"))

# dry run
_ = forward(params, carry, inputs, resets, single=False, seed=0)

# %%

print(f"JAX model has # params: {count_params(params)}")  # 411224064


# %%

# benchmark forward

print("Benchmarking model with forward_compile")
for i in range(10):
  with timer.section("jax_forward_compile"):
    _, out = forward(params, carry, inputs, resets, single=False, seed=0)


# %%

timer.stats()

