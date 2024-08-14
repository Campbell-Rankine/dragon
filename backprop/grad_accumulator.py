import torch as T
from typing import Optional, List, Any

from utils.metrics import average_losses
from utils.torch_utils import Window

"""
TODO:
    - Add the ability to transform the gradient with a list of input functions
    - Implement multiple methods for gradient accumulation other than just the mean
"""


class GradAccumulator:
    def __init__(
        self,
        K: int,
        requires_grad: Optional[bool] = True,
        start_data: Optional[list] = None,
        dtype=T.Tensor,
        torch_dtype=T.float16,  # TODO: pass to torch
        operation=average_losses,
    ):
        if start_data is None:
            start_data = T.tensor([], requires_grad=requires_grad)
        self.N = len(start_data)
        self.K = K
        self.start_data = start_data
        self.window = Window(self.K, self.start_data, operation=operation)
        self.dtype = dtype
        self.operation = operation
        self.requires_grad = requires_grad
        self.torch_dtype = torch_dtype

    def __repr__(self):
        try:
            return f"Gradient Accumulator - Type: {self.dtype}, Max Size: {self.N}, Current Size: {len(self.window.data)}, Data: {self.window.data}"
        except Exception as e:
            print(f"Handling: {e}")
            return f"Gradient Accumulator - Type: {self.dtype}, Max Size: {self.N}, Current Size: {self.window.data.shape[0]}, Data: {self.window.data}"

    def __call__(self, input: Optional[Any] = None):
        # Make call to the window class
        data = self.window(input, requires_grad=self.requires_grad)
        self.N = self.window.N
        return data
