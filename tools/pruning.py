import torch as T
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Tuple, Any, List


class DistinctivenessPruning(prune.BasePruningMethod):

    def __init__(
        self,
        model,
        tolerance: Optional[Tuple[float, float]],
        skip_layers: Optional[List[str]] = [],
        torch_dtype=T.float16,
        device: Optional[T.DeviceObjType] = T.device("cuda:0"),
    ):
        # save attributes
        self.model = model
        self.min_angle = tolerance[0]
        self.max_angle = tolerance[1]
        self.skip_layers = skip_layers
        self.torch_dtype = torch_dtype
        self.device = device

    def update_angle_params(self, val: float, apply_to_min: Optional[bool] = True):
        if apply_to_min:
            self.min_angle = val
        else:
            self.max_angle = val

    def get_angle(self, param1: T.tensor, param2: T.tensor, *args, **kwargs) -> float:
        """
        Calculate angle (degrees) from between weights vectors W_i, W_j
        Args:
          - param1 (T.tensor) : Weight vector 1
          - param2 (T.tensor) : Weight vector 2
          - *args (dict[str, Any]) : Named arguments for Torch.dot
          - **kwargs (dict[str, Any]) : Named arguments for Torch.rad2deg
        """
        numerator = T.dot(param1, param2, *args)
        denominator = T.norm(param1, dtype=self.torch_dtype) * T.norm(
            param2, dtype=self.torch_dtype
        )
        result = T.rad2deg(T.acos(numerator / denominator), **kwargs)
        assert result.device == self.device
        return result

    def merge_neurons(param1: T.tensor, param2: T.tensor, *args, **kwargs):
        raise NotImplementedError

    def delete_neurons(param1: T.tensor, param2: T.tensor, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, iteration: int, *args, **kwargs):
        raise NotImplementedError
