"""
File: utils.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Heavily adapted some modules from Danijar Hafner's implementation
"""

import datetime
import collections
import io
import os
import json
from typing import Dict, Tuple
import pathlib
import re
import time
import random

import numpy as np
import collections
import re
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter

from ...utils import tree


f32 = torch.float32
i32 = torch.int32

COMPUTE_DTYPE = torch.bfloat16
PARAM_DTYPE = torch.float32

"""Convert tensor tree to numpy array tree."""
to_np = lambda x: tree.map(lambda x2: x2.detach().cpu().numpy(), x)

"""Stop gradient - detach tensor from computation graph."""
sg = lambda x: tree.map(lambda x2: x2.detach(), x)

