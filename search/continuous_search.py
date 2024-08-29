import torch as T
import numpy as np
from scipy.optimize import minimize
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ZeroMean
from typing import Optional, List, Tuple, Any, Dict


# internal imports
from search.acquisition.functions import probability_improvement
from search.models.bopt_models import DragonGPR
from search.hyperparam import Hyperparameter


class BaseDragonBayesOpt:
    def __init__(
        self,
        objective_function: callable,
        Y_init: T.tensor,
        hyper_params: List[Hyperparameter],
        acquisition_function: Optional[callable] = probability_improvement,
        kernel: Optional[str] = "Matern5/2",
        eps: Optional[float] = 1e-3,
        iters: Optional[int] = 30,
        optim_fn: Optional[callable] = minimize,
        optim_method: Optional[str] = "L-BFGS-B",
        likelihood: Optional[callable] = GaussianLikelihood(),
        means: Optional[Any] = ZeroMean(),
        kernel_scale_wrapper: Optional[bool] = False,
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
        self.bounds, self.X0 = self.__init_pbounds()
        self._param_length_check(X_init=self.X0)

        self.acquisition = acquisition_function
        self.likelihood = likelihood
        self.regressor = DragonGPR(
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

    def _map_hyperparams_to_obj(self):
        raise NotImplementedError

    def __init_pbounds(self, *tensor_args):
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
            res = minimize(
                fun=min_obj, x0=x0, bounds=self.bounds["vector"].T, method="L-BFGS-B"
            )
            if res.fun < restart_best["value"]:
                restart_best["value"] = res.fun
                restart_best["x"] = res.x
        return restart_best

    def _push_iteration_storage(self, restart_best: dict):
        if -1 * restart_best["value"] >= self.prev_samples["best"]["Y"]:
            self.prev_samples["best"] = {
                "X": restart_best["x"],
                "Y": -1 * restart_best["value"],
            }
        self.prev_samples[f"{self.current_iter}"] = {
            "X": restart_best["x"],
            "Y": -1 * restart_best["value"],
        }

    def fit(self):
        # TODO: Run the entire optimization pipeline (small cost models)
        raise NotImplementedError

    def __call__(
        self,
        model,
        batch_X: T.tensor,
        batch_Y: T.tensor,
        xi: Optional[float] = 0.05,
    ):
        # TODO: Run 1 iteration of the optimization pipeline (high cost models)
        if self.current_iter >= self.iters or self.stop:
            self.__setattr__("stop", True)
            return self.prev_samples["best"]

        # iteration sample
        restart_best = self._sample_next_points(xi=xi)
        print(f"Found new best sample: {restart_best['x']} = {restart_best['value']}")

        # storage handling
        self._push_iteration_storage(restart_best=restart_best)
        x_tensor = T.from_numpy(restart_best["x"])
        print(self._X_sample.shape, x_tensor.shape)
        try:
            self._X_sample = T.concat((self._X_sample, x_tensor), dim=0)
        except:
            x_tensor = x_tensor.unsqueeze(0)
            self._X_sample = T.concat((self._X_sample, x_tensor), dim=0)

        y_tensor = T.tensor([restart_best["value"]])
        try:
            self._Y_sample = T.concat((self._Y_sample, y_tensor), dim=0)
        except:
            y_tensor = y_tensor.unsqueeze(0)
            self._Y_sample = T.concat((self._Y_sample, y_tensor), dim=0)

        # attribute updates
        self.__setattr__("current_iter", self.__getattribute__("current_iter") + 1)

        return restart_best
