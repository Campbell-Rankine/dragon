import torch as T
from typing import Optional


def tensor_app(result: T.Tensor, input, dim: Optional[int] = 0, **kwargs) -> T.tensor:
    """
    Function to append some input to the result of a tensor. Pass the input Tensor args by name into this function
    """
    to_add = T.tensor([input], **kwargs)
    return T.cat((result, to_add), dim=dim)


def tensor_del(tensor, indices):
    mask = T.ones(tensor.numel(), dtype=T.bool)
    mask[indices] = False
    return tensor[mask]
