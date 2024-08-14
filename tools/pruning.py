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
        curr_name, curr_module = self.modules[0]
        next_name, next_module = self.modules[1]

        self.current_module = {
            "curr": (curr_name, curr_module),
            "next": (next_name, next_module),
        }

        self.skip_layers = skip_layers
        self.torch_dtype = torch_dtype
        self.device = device

        # set up trackers for param generation
        self.idx = 0
        self.STOP_FLAG = False

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

    def current_modules(self):
        return self.current_module

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

    def _next_module(self):
        assert self.idx - 2 <= len(self.modules)
        try:
            # current
            name, module = self.modules[self.idx + 1]
            self.idx += 1

            # build data
            self.current_module = {
                "curr": (name, module),
            }
        except IndexError:
            self.STOP_FLAG = True

    def _init_all_module_pairs(self):
        # TODO: Use update_module_data to build out all of the curr, next pairs of modules
        raise NotImplementedError

    def apply(
        self,
        function: callable,
        model: nn.Module,
        over_next_layer: Optional[bool] = False,
        **kwargs,
    ):
        """
        Apply function to the parameters of model
        Args:
        ---
            - function (callable) : Function to determine the pruning weights
            - model (nn.Module) : Model object
            - **kwargs (dict[str, Any]) : Named arguments for the pruning weight function
        """
        # TODO: write function to iterate and move across all module parameters.
        result = {}
        while self.STOP_FLAG == False:
            # initial objects
            param_finished = False

            # check pair against ground truth
            m = self.current_modules()
            name, module = m["curr"]

            if not name in result.keys():
                result[name] = []

            # retrieve module params
            with T.no_grad():
                for idx, (name_, wi) in enumerate(module.named_parameters()):
                    if "weight" in name_:
                        new_weights = None
                        wj = None
                        if over_next_layer:
                            try:
                                wj = next(module.named_parameters())
                            except Exception as e:
                                print(e)
                                break
                            new_weights = function(wi, wj, model, **kwargs)
                        else:
                            new_weights = function(wi, model, **kwargs)

                        # add to result

                        wi.copy_(new_weights)
                        result[name].append({name_: new_weights})

            # update module
            self._next_module()
        return result


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
        raise NotImplementedError
