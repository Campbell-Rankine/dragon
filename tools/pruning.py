import torch as T
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Tuple, Any, List


# class IterativeDragonPruner: (for applying a pruning function iteratively, abstract class)
class IterativeDragonPruner(prune.BasePruningMethod):
    def __init__(
        self,
        model: nn.Module,
        skip_layers: Optional[List[str]] = [],
        torch_dtype=T.float16,
        device: Optional[T.DeviceObjType] = T.device("cuda:0"),
    ):
        super().__init__()
        self.modules = list(model.named_modules())[
            1:
        ]  # first module is the base class with no name
        self.skip_layers = skip_layers
        self.torch_dtype = torch_dtype
        self.device = device

    # attributes
    @property
    def allowed_modules(self):
        return [
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.RNN,
            nn.LSTM,
            nn.GRU,
            nn.Transformer,
        ]

    # functions
    def _check_module_instance(self, module: list[nn.Module] | nn.Module):

        def module_instance_check(module: nn.Module, allowed) -> bool:
            for instance in allowed:
                if isinstance(module, instance):
                    return True
            return False

        if type(module) == list:
            for mod in module:
                assert module_instance_check(module=mod, allowed=self.allowed_modules)
        else:
            assert module_instance_check(module=mod, allowed=self.allowed_modules)

    def compute_mask(self):
        raise NotImplementedError

    def _get_next_module(self, module_name: str) -> str:
        return_flag = False
        for name, module in self.modules:
            if return_flag:
                return name, module
            if name == module_name:
                return_flag = True

    def current_modules(self):
        for (
            name,
            module,
        ) in (
            self.modules
        ):  # this is how you iterate through the named modules "" for the base. lstm for LSTM layer and fc2 for FC layer
            if name == "":
                continue
            else:
                self.current_module = name
                next_name, next_module = self._get_next_module(name)
                self._check_module_instance([module, next_module])
                return (
                    {name: module},
                    {next_name: next_module},
                )  # TODO: Change this to include the modules as well.

    def prune_iterative_callable(self, model: nn.Module, function: callable):
        # TODO: write function to iterate and move across all module parameters.
        for name, module in model.named_modules():
            if name in self.skip_layers:
                continue


# class BaseDragonPruner: (for applying some kind of default mask)

"""
TODO:
    - Implement model parameter init function for merge/delte
    - Double check the rest of the abstract methods
    - test on a transformer, CNN and LSTM
"""


class DistinctivenessPruning(IterativeDragonPruner):
    """
    Prune a pytorch model of the types allowed by IterativeDragonPruner (extends IterativeDragonPruner). Prunes weights by either merging or deleting weight vectors that point the same / opposite directions.
    If the angle between weight vectors W_i and W_j < 30 they are more or less contributing the same thing, so average the weights and combine them into a single vector W_r.
    If the angle between weight vectors W_i and W_j > 150 they are more or less cancelling eachother out. So delete both.
    """

    def __init__(
        self,
        tolerance: Optional[Tuple[float, float]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.current_module = self.modules[0]

        # init attributes
        self.min_angle = tolerance[0]
        self.max_angle = tolerance[1]

    def get_angle(self, param1: T.tensor, param2: T.tensor, *args, **kwargs) -> float:
        """
        Calculate angle (degrees) from between weights vectors W_i, W_j
        Args:
        ---
          - param1 (T.tensor) : Weight vector 1
          - param2 (T.tensor) : Weight vector 2
          - *args (dict[str, Any]) : Named arguments for Torch.dot
          - **kwargs (dict[str, Any]) : Named arguments for Torch.norm
        """
        numerator = T.dot(param1, param2, *args)
        denominator = T.norm(param1, **kwargs) * T.norm(param2, **kwargs)
        result = T.rad2deg(T.acos(numerator / denominator))
        return result.to(self.device)

    def merge_neurons(self, param1: T.tensor, param2: T.tensor, *args, **kwargs):
        raise NotImplementedError

    def delete_neurons(self, param1: T.tensor, param2: T.tensor, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, model: nn.Module, iteration: int, *args, **kwargs):
        for name, module in list(
            model.named_modules()
        ):  # this is how you iterate through the named modules "" for the base. lstm for LSTM layer and fc2 for FC layer
            if name == "":
                continue
            else:
                self.current_module = name
                next_module = self._get_next_module(model, name)
                print(name, next_module)
