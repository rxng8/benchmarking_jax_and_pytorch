# %%


# using `amin` conda environment. python=3.11 and pytorch with cuda 12.6
import sys, pathlib

import torch
from torch import nn
from amin.models.modules import GRU, MLP
from amin.utils import timer

def count_params(module: torch.nn.Module) -> int:
  return sum(p.numel() for p in module.parameters())


device = torch.device("cuda")

class Model(nn.Module):
  def __init__(self, in_units, out_units, hidden):
    super().__init__()
    self.gru = GRU(in_unit=in_units, out_unit=hidden)
    self.mlp = MLP(in_units=hidden, units=out_units, layers=20)

  def initial(self, batch_size):
    return self.gru.initial(batch_size)

  def forward(self, carry, x, reset, single: bool = True):
    carry, h = self.gru(carry, x, reset, single=single)
    y = self.mlp(h)
    return carry, y

# config
I = 2048 # input dim
O = 4096 # output dim
Z = 4096 # hidden dim
S = 1024 # state dim
B = 32 # batch size
T = 64 # sequence length


model = Model(in_units=I, out_units=O, hidden=Z).to(device)
model_compile = Model(in_units=I, out_units=O, hidden=Z).to(device)
model_compile.compile()
carry = model.initial(B).to(device)
inputs = torch.randn(B, T, I).to(device)
resets = torch.zeros(B, T, dtype=torch.bool).to(device)

@torch.compile
def forward_compile(carry, inputs, resets):
  carry, y = model(carry, inputs, resets, single=False)
  return carry, y

@torch.compile
def forward_compile_with_model(carry, inputs, resets):
  carry, y = model_compile(carry, inputs, resets, single=False)
  return carry, y

def forward_eager(carry, inputs, resets):
  carry, y = model(carry, inputs, resets, single=False)
  return carry, y


timer.reset()
_ = forward_compile(carry, inputs, resets)  # warmup
_ = forward_compile_with_model(carry, inputs, resets)  # warmup

# %%

count_params(model)  # 411312128


# %%

print("Benchmarking model with forward_compile")
for i in range(10):
  with timer.section("forward_compile"):
    forward_compile(carry, inputs, resets)


print("Benchmarking model_compile with forward_compile_with_model")
for i in range(10):
  with timer.section("forward_compile_with_model_compile"):
    forward_compile_with_model(carry, inputs, resets)

print("Benchmarking model with forward_eager")
for i in range(10):
  with timer.section("forward_eager"):
    forward_eager(carry, inputs, resets)


timer.stats(reset=False)


