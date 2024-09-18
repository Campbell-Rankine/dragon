import torch as T
from typing import Optional, Tuple, Any, List, Literal, Iterable


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
