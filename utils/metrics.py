"""
Metrics function definitions for tools.logging
"""

import torch as T
import torch.nn as nn
import os
import numpy as np
from typing import Optional, Tuple, Any, List, Literal, Iterable
from utils.torch_utils import tensor_app, tensor_del


def gradient_norm(model: nn.Module, type: Optional[str] = "") -> float:
    # Calculate the norm (L1/L2/Frobenius)
    total_norm: float = 0.0
    norm_input: str | float | None = "fro"  # more of a note of what the default is
    pow: float = 1.0
    match type:
        case "L1":
            norm_input = 1.0
        case "L2":
            norm_input = 2.0
            pow == 0.5
        case "fro":
            norm_input = None
            pow = 0.5
        case _:
            raise ValueError(f"Invalid Grad Norm type of: {type}")

    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(norm_input)
        total_norm += param_norm

    return total_norm**pow


def average_losses(
    data,
    mean: Optional[callable] = T.mean,
    object: Optional[Iterable] = T.tensor,
    **kwargs,
) -> float:
    if not type(data) == object:
        data = object(data, **kwargs)
    return mean(data)
