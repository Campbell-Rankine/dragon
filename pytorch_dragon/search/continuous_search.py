import torch as T
import numpy as np
from scipy.optimize import minimize
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ZeroMean
from typing import Optional, List, Tuple, Any, Dict


# internal imports
from acquisition.functions import probability_improvement, expected_improvement
from search.models.bopt_models import DragonGPR, BanditosGPR
from search.hyperparam import Hyperparameter


class BaseDragonBayesOpt:
    def __init__(
        self,
        objective_function: callable,
        Y_init: T.tensor,
        hyper_params: List[Hyperparameter],
        acquisition_function: Optional[callable] = expected_improvement,
        kernel: Optional[str] = "Matern5/2",
        eps: Optional[float] = 1e-3,
        iters: Optional[int] = 50,
        optim_fn: Optional[callable] = minimize,
        optim_method: Optional[str] = "COBYQA",
        likelihood: Optional[callable] = GaussianLikelihood(),
        means: Optional[Any] = ZeroMean(),
        kernel_scale_wrapper: Optional[bool] = True,
        max_optim_iters: Optional[int] = None,
        regressor_type: Optional[str] = "base",
    ):
        # init attributes
        self.obj_fnc = objective_function
        self.kernel_str = kernel
        self.eps = eps
        self.iters = iters
        self.current_iter = 0
        self.opt_fn = optim_fn
        self.opt_m = optim_method
        self.params = hyper_params
        self.bounds, self.X0 = self.__init_pbounds(regressor_type)
        self._param_length_check(X_init=self.X0)

        self.acquisition = acquisition_function
        self.likelihood = likelihood
        self.regressor = None
        self.regressor_type = regressor_type
        if regressor_type == "base":
            self.regressor = DragonGPR(
                self.X0,
                Y_init,
                likelihood=self.likelihood,
                means=means,
                kernel=self.kernel_str,
                scale_wrapper=kernel_scale_wrapper,
                alpha=eps,
            )
        if regressor_type == "banditos":
            self.regressor = BanditosGPR(
                self.X0,
                Y_init,
                likelihood=self.likelihood,
                means=means,
                kernel=self.kernel_str,
                scale_wrapper=kernel_scale_wrapper,
                alpha=eps,
            )

        # init storage
        self.Y0 = Y_init
        if not len(self.X0.shape) > 1:
            self._X_sample = self.X0.unsqueeze(0)
        self._X_sample = self.X0
        self._Y_sample = self.Y0

        self.prev_samples = {"best": {"X": self.X0, "Y": self.Y0}}
        self.stop = False
        self.max_optim_iters = max_optim_iters

    def _param_length_check(self, X_init):
        try:
            self.N = len(self.params)
            if not len(self.params) == X_init.size(-1):
                raise ValueError(
                    f"Input shape for Hyperparams: {len(self.params)} does not match X_init shape: {X_init.size(-1)}"
                )
        except IndexError:
            self.N = 1
            if not len(self.params) == 1:
                raise ValueError(
                    f"Input shape for Hyperparams: {len(self.params)} does not match X_init shape: {1}"
                )

    def __init_pbounds(self, regressor_type: str, *tensor_args):
        constraint_fn_app = {}
        values = []
        names = []
        for x in self.params:
            names.append(x.name)
            constraint_fn_app[x.name] = [x.range]
            values.append(x.value)
        constraint_fn_app["vector"] = np.vstack(
            (
                [constraint_fn_app[key][0].min for key in names],
                [constraint_fn_app[key][0].max for key in names],
            )
        )
        self.param_names = names
        return constraint_fn_app, T.tensor(values, *tensor_args)

    def _sample_from_bounds(
        self,
        sampling: Optional[callable] = None,
        num_restarts: Optional[int] = 25,
        **kwargs,
    ):
        """
        Sample from acquisition distribution. Optional sampling override function in arguments
        """
        if not sampling is None:
            return sampling(**kwargs)
        else:
            vector = self.bounds["vector"].T
            sample = np.random.uniform(
                vector[:, 0], vector[:, 1], size=(num_restarts, self.N)
            )
            return sample

    def _sample_next_points(self, xi, **kwargs):
        """
        Use GPR and optimizer function/method to sample the most likely points at the maximum of objective function
        Args:
            - xi (float) : Exploration vs exploitation parameter for acquisition function
            - **kwargs (Dict[str, Any]) : acquisition function named arguments
        """
        restart_best: dict = {"value": 0.0, "x": None}

        def min_obj(x: np.ndarray):
            assert type(x) == np.ndarray
            return -self.acquisition(
                x.reshape(-1, self.N), self._X_sample, self.regressor, xi
            )

        bound_samples = self._sample_from_bounds()
        for x0 in bound_samples:
            # handle num_iters
            if self.max_optim_iters is None:
                res = minimize(
                    fun=min_obj,
                    x0=x0,
                    bounds=self.bounds["vector"].T,
                    method=self.opt_m,
                )
            else:
                res = minimize(
                    fun=min_obj,
                    x0=x0,
                    bounds=self.bounds["vector"].T,
                    method=self.opt_m,
                    maxiters=self.max_optim_iters,
                )
            if res.fun < restart_best["value"]:
                restart_best["value"] = res.fun
                restart_best["x"] = res.x
        return restart_best

    def _push_iteration_storage(self, restart_best: dict):
        best_val = -1 * restart_best["value"]
        if self.regressor_type == "banditos":
            best_val = np.exp(best_val)
        if -1 * restart_best["value"] >= self.prev_samples["best"]["Y"]:
            # push to best
            self.prev_samples["best"] = {
                "X": restart_best["x"],
                "Y": best_val,
            }
        # store history
        self.prev_samples[f"{self.current_iter}"] = {
            "X": restart_best["x"],
            "Y": best_val,
        }

    def add_to_x(self, x_tensor):
        try:
            self._X_sample = T.concat((self._X_sample, x_tensor), dim=0)
        except:
            x_tensor = x_tensor.unsqueeze(0)
            self._X_sample = T.concat((self._X_sample, x_tensor), dim=0)

    def add_to_y(self, y_tensor):
        try:
            self._Y_sample = T.concat((self._Y_sample, y_tensor), dim=0)
        except:
            y_tensor = y_tensor.unsqueeze(0)
            self._Y_sample = T.concat((self._Y_sample, y_tensor), dim=0)

    def assign_values_to_hyperparams(self, input_tensor: T.tensor):
        assert input_tensor.shape[0] == len(self.params)
        for i in range(len(self.params)):
            value = input_tensor[i].item()
            if self.regressor_type == "banditos":
                value = np.exp(value)
            self.params[i].assign(value)

    def __call__(  # TODO: assert no gradients being calculated here. Use as little of the gpu as possible, maybe even convert to cpu
        self,
        model,
        batch_X: T.tensor,
        batch_Y: T.tensor,
        xi: Optional[float] = 0.05,
        **kwargs,
    ):
        """
        Run a bayes opt iteration. Get model output, sample most likely hyperparameters to maximize the objective function, return best params.
        Args:
            - model (Any) : Pytorch model object. Currently only supports nn.Module (call)
            - batch_X (T.tensor) : Input data to the model
            - batch_Y (T.tensor) : Model ground truth
            - xi (float) : Optional exploitation vs exploration parameter (default: 0.05)
            - **kwargs : Objective function named arguments
        """
        if self.current_iter >= self.iters or self.stop:
            self.__setattr__("stop", True)
            return self.prev_samples["best"]

        # iteration sample
        restart_best = self._sample_next_points(xi=xi)
        best = -1 * restart_best["value"]
        if self.regressor_type == "banditos":
            best = np.exp(best)

        # storage handling
        self._push_iteration_storage(restart_best=restart_best)

        # push new samples to internal storage (self._X_sample, self._Y_sample)
        x_tensor = T.from_numpy(restart_best["x"])
        y_tensor = self.obj_fnc(model, batch_X, batch_Y, **kwargs)
        self.add_to_x(x_tensor.exp())
        self.add_to_y(y_tensor.exp())

        # attribute updates
        self.__setattr__("current_iter", self.__getattribute__("current_iter") + 1)
        self.assign_values_to_hyperparams(x_tensor)

        return restart_best
