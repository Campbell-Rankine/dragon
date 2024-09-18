"""
PyTorch define hyperparameter types
"""

import torch as T
import numpy as np
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
        return result


class RangeConstraint:
    def __init__(self, max_: float, min_: float, log_range: Optional[bool] = False):
        self.max = max_
        self.min = min_
        self.log_range = log_range

    def __call__(self, x: Any):
        if x > self.max:
            return False
        if x < self.min:
            return False
        return True

    def __repr__(self):
        return f"Max: {self.max}, Min: {self.min}"


class Hyperparameter:
    def __init__(
        self,
        name: str,
        type_: str,
        x: Optional[Any] = 0.0,
        constraints: Optional[List[Constraint]] = [],
        range_: Optional[Tuple[Any, Any]] = (0.0, 1.0),
        sampling_fn: Optional[callable] = T.rand,
        tensor_kwargs: Optional[Dict[str, Any]] = None,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = "cpu",
        cache_future_values: Optional[
            bool
        ] = False,  # TODO: Implement Caching of future values
        cache_window: Optional[int] = 30,
        log_range: Optional[bool] = False,
    ):
        # initial attribute setup
        self.name = name
        self.type = type_
        self.constraints = constraints
        self.range = RangeConstraint(
            max_=range_[1], min_=range_[0], log_range=log_range
        )
        self.__device = device
        self.sampler = sampling_fn
        self.future_cache = cache_future_values
        self.sample_kwargs = sample_kwargs
        self.log_range = log_range

        self.__storage = {"previous": [], "cache": []}

        # Assign value
        if tensor_kwargs is None:
            tensor_kwargs = {}

        # value init check
        try:
            x_ = 0.0
            if log_range:
                x_ = np.log(x)
            assert self.range(x)
            self.value = T.tensor(x_, device=self.__device, **tensor_kwargs)
        except AssertionError:
            raise ValueError(
                f"Initial value for {self.name}={x} does not satisfy range constraint -> {self.range}"
            )

        # Constraints
        if not constraints is None:
            self.apply_constraints()

        assert self._type_check()

    def assign(self, val: Any, **kwargs):
        self.__storage["previous"].append(
            self.__getattribute__("value")
        )  # append to storage
        self.__setattr__("value", T.tensor(val, device=self.__device, **kwargs))
        if not self.apply_constraints():
            raise ValueError("val does not satisfy constraints")

    def update_sample_args(self, args: Dict[str, Any]):
        self.__setattr__("sample_kwargs", args)

    def __iter__(self, **tensor_kwargs):
        if self.sample_kwargs is None:
            self.sample_kwargs = {}
        next_val = self.sampler(**self.sample_kwargs)

        self.assign(next_val, **tensor_kwargs)

        return self.item

    def __repr__(self):
        val = self.value
        if self.log_range:
            val = np.exp(self.value)
        return f"{self.name}={val}"

    def apply_constraints(self, **kwargs):
        val = False
        if self.__getattribute__("constraints") == []:
            return True
        for x in self.__getattribute__("constraints"):
            try:
                val = x(self.value, **kwargs)
            except AssertionError:
                return False
        return val

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
