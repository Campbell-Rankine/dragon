"""
PyTorch define hyperparameter types
"""

import torch as T
from typing import Optional, List, Tuple, Any, Dict


class Constraint:
    def __init__(self, fn: callable):
        self.fn = fn

    def __call__(self, x: Any, **kwargs):
        result = self.fn(x, **kwargs)
        if not type(result) == bool:
            raise ValueError(
                "Invalid constraint function, please make sure the function returns a boolean"
            )
        assert result


class Hyperparameter:
    def __init__(
        self,
        name: str,
        type_: str,
        x: Optional[Any] = 0.0,
        constraints: Optional[List[Constraint]] = None,
        range_: Optional[Tuple[Any, Any]] = (0.0, 1.0),
        sampling_fn: Optional[callable] = T.rand,
        tensor_kwargs: Optional[Dict[str, Any]] = None,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = "cpu",
        cache_future_values: Optional[bool] = False,
    ):
        # initial attribute setup
        self.name = name
        self.type = type_
        self.contraints = constraints
        self.range = range_
        self.__device = device
        self.sampler = sampling_fn
        self.future_cache = cache_future_values
        self.sample_kwargs = sample_kwargs

        self.__storage = {"previous": [], "cache": []}

        # Assign value
        if tensor_kwargs is None:
            tensor_kwargs = {}
        self.value = T.tensor(x, device=self.__device, **tensor_kwargs)

        # Constraints
        if not constraints is None:
            self._apply_constraints()

        assert self._type_check()

    def assign(self, val: Any, **kwargs):
        self.__storage["previous"].append(
            self.__getattr__("value")
        )  # append to storage
        self.__setattr__("value", T.tensor(val, device=self.__device, **kwargs))

    def update_sample_args(self, args: Dict[str, Any]):
        self.__setattr__("sample_kwargs", args)

    def __iter__(self, **tensor_kwargs):
        if self.sample_kwargs is None:
            self.sample_kwargs = {}
        next_val = self.sampler(**self.sample_kwargs)

        self.assign(next_val, **tensor_kwargs)

        return self.item

    # helper functions
    @property
    def device(self):
        return self.__device

    @property
    def item(self):
        return self.value.item()

    @property
    def search_type(self):
        return self.type

    @property
    def history(self):
        return self.__storage["previous"]

    def __repr__(self):
        return f"{self.name}={self.value}"

    def _apply_constraints(self, **kwargs):
        for x in self.constraints:
            assert isinstance(x, callable)
            x(**kwargs)

    def _type_check(self):
        match self.type:
            case "discrete":
                return True
            case "Discrete":
                return True
            case "numerical":
                return True
            case "Numerical":
                return True
            case "model":
                return True
            case "Model":
                return True
            case _:
                raise ValueError(
                    f"Invalid hyperparameter search type: {self.type} not supported"
                )
