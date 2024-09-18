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
            nn.ConvTranspose1d,
            nn.RNN,
            nn.LSTM,
            nn.GRU,
            nn.Transformer,
        ]

    @property
    def allowed_modules_str(self):
        return [
            "linear",
            "conv1d",
            "rnn",
            "lstm",
            "gru",
            "transformer",
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

    def modify_weight(self, model, name, value):
        model.__getattr__(name).__setattr__("weight", value)
        return model

    def _init_all_module_pairs(self):
        # TODO: Use update_module_data to build out all of the curr, next pairs of modules. Low priority as will be useful in runtime optimization
        raise NotImplementedError

    def apply_to(
        self,
        function: callable,
        model: nn.Module,
        name: str,
        idx: int,
        over_next_layer=False,
    ):

        result = {}
        result[name] = []
        # retrieve module params
        with T.no_grad():
            for index, (name_, module) in enumerate(model.named_parameters()):
                if "weight" in name_ and name_ == name:
                    for idx_, param in enumerate(module[:-1]):
                        if idx == idx_:
                            wi = param
                            wj = None
                            if over_next_layer:
                                wj = module[idx + 1]
                                new_weights_i, new_weights_j = function(wi, wj)
                                wi.copy_(new_weights_i)
                                wj.copy_(new_weights_j)
                                result[name].append(
                                    {name_ + "_" + str(idx): new_weights_i}
                                )
                                result[name].append(
                                    {name_ + "_" + str(idx + 1): new_weights_j}
                                )
                            else:
                                new_weights = function(wi)
                                wi.copy_(new_weights)
                                result[name].append({name_: new_weights})
        return result


class DistinctivenessPruning(IterativeDragonPruner):
    """
    Prune a pytorch model of the types allowed by IterativeDragonPruner (extends IterativeDragonPruner). Prunes weights by either merging or deleting weight vectors that point the same / opposite directions.
    If the angle between weight vectors W_i and W_j < 30 they are more or less contributing the same thing, so average the weights and combine them into a single vector W_r.
    If the angle between weight vectors W_i and W_j > 150 they are more or less cancelling eachother out. So delete both.

    # TODO: Cleanup above class
    # TODO: Cleanup this class
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
        self.mask = {"deleted": [], "merged": []}  # Only push merged neurons to remove.

    def compute_mask(self, t, default_mask) -> T.Tensor:
        """
        Compute mask based on importance scores t
        Args:
            - t (T.tensor) : Importance scores
            - default_mask (T.Tensor) : Pytorch base mask
        Returns:
            T.Tensor
        """
        raise NotImplementedError

    def get_angle(self, param1: T.tensor, param2: T.tensor, *args, **kwargs) -> float:
        """
        Calculate angle (degrees) from between weights vectors W_i, W_j
        Args:
            - param1 (T.tensor) : Weight vector 1
            - param2 (T.tensor) : Weight vector 2
            - *args (dict[str, Any]) : Named arguments for Torch.dot
            - **kwargs (dict[str, Any]) : Named arguments for Torch.norm
        Returns:
            T.tensor
        """
        numerator = T.dot(param1, param2, *args)
        denominator = T.norm(param1, **kwargs) * T.norm(param2, **kwargs)
        result = T.rad2deg(T.acos(numerator / denominator))
        return result.to(self.device)

    def merge_neurons(
        self,
        param1: T.tensor,
        param2: T.tensor,
    ) -> Tuple[T.tensor, T.tensor]:
        """
        Modify gradients for param1 := 1/2 (param1+param2). Must return averaged weights, T.zeros_like(param2) to comply with IterativeDragonPruner
        Args:
            - param1 (T.tensor) : Weight vector 1
            - param2 (T.tensor) : Weight vector 2
            - weights (Tuple[float, float]) : Weights for the weighted sum. Default value averages the weights
            - **kwargs (Dict[str, Any]) : Named arguments to pass to T.zeros_like()
        Returns:
            Tuple[T.tensor, T.tensor]
        """
        # get result tensor

        result = (0.5 * param1) + (0.5 * param2)
        return result, T.zeros_like(param2)

    def _prune_parameter(
        self,
        name: str,
        model: nn.Module,
        **kwargs,
    ):
        """
        Apply the correct pruning function to each of the modules. Store the parameters to be pruned, and store the values to calculate the mask
        Args:
            - name (str) : Name of the model parameter to prune
            - model (nn.Module) : Pytorch Module
            - over_next_layer (Optional[bool]) : Apply to two layers of weight parameters
            - **kwargs (Dict[str, Any]) : Keyword args for self.apply_to

        """
        result_ = {}
        result = {}
        result["angle"] = []
        result[name] = []
        # retrieve module params
        with T.no_grad():
            for idx, (name_, module) in enumerate(model.named_parameters()):
                if "weight" in name_ and name_ == name:
                    old_weights = module
                    for idx, param in enumerate(module[:-1]):
                        # get params
                        wvi = param
                        wvj = module[idx + 1]

                        # calc angle and apply tolerance
                        angle = self.get_angle(wvi, wvj)
                        result["angle"].append(angle)
                        if angle <= self.min_angle:
                            res = self.apply_to(
                                self.merge_neurons,
                                model,
                                name,
                                idx,
                                **kwargs,
                            )
                            result[name] = res[name]
                            self.mask["merged"].append((name, name_, idx + 1))
                        if angle >= self.max_angle:
                            self.mask["deleted"].append((name, name_, idx))
                            self.mask["deleted"].append((name, name_, idx + 1))
                    ind = None
                    new_weights = None
                    try:
                        ind = int(list(result[name][0].keys())[0].split("_")[-1])
                    except IndexError:
                        continue
                    new_weights = dict(model.named_parameters())[name]
                    new_weights[ind] = result[name][0][f"{name}_{ind}"]
                    new_weights[ind + 1] = T.zeros_like(new_weights[ind])
        if not new_weights is None:
            model.__getattr__(name.split(".")[0]).__setattr__("weight", new_weights)
        return result, model

    def __call__(
        self,
        parameter: str,
        model: nn.Module,
        over_next_layer: Optional[bool] = False,
        return_result_dict: Optional[bool] = False,
        **kwargs,
    ):
        """
        Run distinctiveness pruning across all model parameters
        """
        layer_type = ""
        param = parameter.split(".")[0].lower()

        match param:
            case s if "fc" in s:
                layer_type = "linear"
            case s if "lstm" in s:
                layer_type = "lstm"
            case s if "gru" in s:
                layer_type = "gru"
            case s if "rnn" in s:
                layer_type = "rnn"
            case s if "transformer" in s:
                layer_type = "transformer"

        if not layer_type in self.allowed_modules_str:
            raise ValueError
        result, model_ = self._prune_parameter(
            parameter, model, over_next_layer=over_next_layer
        )
        if return_result_dict:
            return model_, result
        return model_, None
