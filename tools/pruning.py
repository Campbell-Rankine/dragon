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
        increment: Optional[int] = 1,
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
        self.inc = increment

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
            name, module = self.modules[self.idx + self.inc]
            self.idx += self.inc

            # build data
            self.current_module = {
                "curr": (name, module),
            }
        except IndexError:
            self.STOP_FLAG = True

    def _init_all_module_pairs(self):
        # TODO: Use update_module_data to build out all of the curr, next pairs of modules. Low priority as will be useful in runtime optimization
        raise NotImplementedError

    def apply_to(
        self,
        function: callable,
        model: nn.Module,
        name: str,
        over_next_layer=False,
        **kwargs,
    ):
        # TODO: write function to only apply callable to one specific named_param
        result = {}
        result[name] = []
        # retrieve module params
        with T.no_grad():
            for idx, (name_, wi) in enumerate(model.named_parameters()):
                if "weight" in name_ and name_ == name:
                    new_weights = None
                    wj = None
                    next_name = None
                    if (
                        over_next_layer
                    ):  # I reckon get rid of this functionality, you can simply
                        try:
                            next_name, wj = next(
                                model.named_parameters()
                            )  # TODO: Test 2 parameter pruning
                        except Exception as e:
                            print(e)
                            break
                        new_weights_i, new_weights_j = function(wi, wj, model, **kwargs)
                        wi.copy_(new_weights_i)
                        wj.copy_(new_weights_j)
                        result[name].append({name_: new_weights_i})
                        result[name].append({next_name: new_weights_j})
                    else:
                        new_weights = function(wi, model, **kwargs)
                        wi.copy_(new_weights)
                        result[name].append({name_: new_weights})
        return result

    def apply_fn(
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
                        next_name = None
                        if (
                            over_next_layer
                        ):  # I reckon get rid of this functionality, you can simply
                            try:
                                next_name, wj = next(
                                    module.named_parameters()
                                )  # TODO: Test 2 parameter pruning
                            except Exception as e:
                                print(e)
                                break
                            new_weights_i, new_weights_j = function(
                                wi, wj, model, **kwargs
                            )
                            wi.copy_(new_weights_i)
                            wj.copy_(new_weights_j)
                            result[name].append({name_: new_weights_i})
                            result[name].append({next_name: new_weights_j})
                        else:
                            new_weights = function(wi, model, **kwargs)
                            wi.copy_(new_weights)
                            result[name].append({name_: new_weights})

            # update module
            self._next_module()
        return result


class DistinctivenessPruning(IterativeDragonPruner):
    """
    Prune a pytorch model of the types allowed by IterativeDragonPruner (extends IterativeDragonPruner). Prunes weights by either merging or deleting weight vectors that point the same / opposite directions.
    If the angle between weight vectors W_i and W_j < 30 they are more or less contributing the same thing, so average the weights and combine them into a single vector W_r.
    If the angle between weight vectors W_i and W_j > 150 they are more or less cancelling eachother out. So delete both.
    """

    def __init__(
        self,
        tolerance: Optional[Tuple[float, float]],
        save_grad: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init attributes
        self.min_angle = tolerance[0]
        self.max_angle = tolerance[1]
        self._save_grad = save_grad

        # storage
        self.mask = []

    def compute_mask(self, t, default_mask) -> T.Tensor:
        """
        Compute mask based on importance scores t
        Args:
            - t (T.tensor) : Importance scores
            - default_mask (T.Tensor) : Pytorch base mask
        Returns:
            - mask (T.Tensor) : mask.shape == t.shape
        """
        raise NotImplementedError

    def get_angle(self, param1: T.tensor, param2: T.tensor, *args, **kwargs) -> float:
        """
        Calculate angle (degrees) from between weights vectors W_i, W_j
        Args:
        ---
            - param1 (T.tensor) : Weight vector 1
            - param2 (T.tensor) : Weight vector 2
            - *args (dict[str, Any]) : Named arguments for Torch.dot
            - **kwargs (dict[str, Any]) : Named arguments for Torch.norm
        Returns:
            result (T.tensor) - resultant angle
        """
        numerator = T.dot(param1, param2, *args)
        denominator = T.norm(param1, **kwargs) * T.norm(param2, **kwargs)
        result = T.rad2deg(T.acos(numerator / denominator))
        return result.to(self.device)

    def merge_neurons(
        self,
        param1: T.tensor,
        param2: T.tensor,
        weights: Optional[Tuple[float, float]] = (0.5, 0.5),
        **kwargs,
    ) -> Tuple[T.tensor, T.tensor]:
        """
        Modify gradients for param1 := 1/2 (param1+param2). Must return averaged weights, T.zeros_like(param2) to comply with IterativeDragonPruner
        Args:
        ---
            - param1 (T.tensor) : Weight vector 1
            - param2 (T.tensor) : Weight vector 2
            - weights (Tuple[float, float]) : Weights for the weighted sum. Default value averages the weights
            - **kwargs (Dict[str, Any]) : Named arguments to pass to T.zeros_like()
        Returns:
            - wvi, wvj (Tuple[T.tensor, T.tensor]) : Modified weights for param1, param2
        """
        # get result tensor
        assert weights[0] + weights[1] == 1
        result = (weights[0] * param1) + (weights[1] * param2)
        return result, T.zeros_like(param2, **kwargs)

    def _prune_parameter(self, name: str, model: nn.Module, **kwargs):
        result = {}
        result[name] = []
        # retrieve module params
        with T.no_grad():
            for idx, (name_, wi) in enumerate(model.named_parameters()):
                if "weight" in name_ and name_ == name:
                    raise NotImplementedError

    def __call__(
        self,
        model: nn.Module,
        parameter: str,
        use_nograd: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        """
        Run distinctiveness pruning across all model parameters
        """
        raise NotImplementedError
