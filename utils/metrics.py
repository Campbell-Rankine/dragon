"""
Metrics function definitions for tools.logging
"""

import torch as T
import torch.nn as nn
import os
import numpy as np
from typing import Optional, Tuple, Any, List, Literal, Iterable
from utils.torch_utils import tensor_app, tensor_del


class Window:
    def __init__(
        self,
        size,
        initial_data: Optional[List] = [],
        operation: Optional[callable] = None,
    ):
        self.data = initial_data
        self.N = len(self.data)
        self.K = size
        self.operation = operation

    def push_to_stack(self, item) -> bool:
        try:
            self.data.append(item)
            self.data.remove(self.data[0])
            assert len(self.data) <= self.K - 1
            assert self.n + 1 <= self.k
            self.n += 1
            return True
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def __call__(
        self,
        input: Optional[Any] = None,
        append_function: Optional[callable] = tensor_app,
        delete_function: Optional[callable] = tensor_del,
        requires_grad: Optional[bool] = False,
        **kwargs,
    ) -> Any:
        assert not (self.N > self.K)
        if input is None:
            if not self.operation is None:
                return self.operation(self.data, **kwargs)
            else:
                return self.data
        try:
            self.data = append_function(self.data, input, requires_grad=requires_grad)

            self.N += 1
            if self.N > self.K:
                self.data = delete_function(
                    self.data, [0]
                )  # args: object, index to del
                self.N -= 1

            if not self.operation is None:
                return self.operation(self.data, **kwargs)
            else:
                return self.data
        except Exception as e:
            raise e


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
