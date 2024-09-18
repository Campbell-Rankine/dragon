"""
Module for gradient penalties to prevent vanishing/exploding gradients

TODO:
  - Implement / Test l1
  - Implement / Test l2
  - Research, implement, and test a third method
"""

import torch as T
import torch.nn as nn
from typing import Optional


class L1(nn.Module):
    def __init__(self, eps: Optional[float] = 1e-3):
        self.eps = eps

    def __call__(self, grads: T.tensor, **kwargs):
        raise NotImplementedError


class L2(nn.Module):
    def __init__(self, eps: Optional[float] = 1e-3):
        self.eps = eps

    def __call__(self, grads: T.tensor, **kwargs):
        raise NotImplementedError


class LI(nn.Module):  # TODO: research what this module should be
    def __init__(self, eps: Optional[float] = 1e-3):
        self.eps = eps

    def __call__(self, grads: T.tensor, **kwargs):
        raise NotImplementedError
