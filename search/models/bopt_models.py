import torch as T
import numpy as np
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from botorch.posteriors import GPyTorchPosterior, HigherOrderGPPosterior
from gpytorch.means import ZeroMean
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from typing import Optional, Any

from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


class DragonGPR(ExactGP):
    def __init__(
        self,
        train_x: T.tensor,
        train_y: T.tensor,
        likelihood: GaussianLikelihood,
        means: Optional[Any] = ZeroMean(),
        kernel: Optional[str] = "Matern5/2",
        scale_wrapper: Optional[bool] = True,
        alpha: Optional[float] = 1e-3,
    ):
        super().__init__(
            train_inputs=train_x, train_targets=train_y, likelihood=likelihood
        )
        self.mean_fn = means
        self.kernel = self.__init_kernel(kernel)
        self.noise = alpha
        if scale_wrapper:
            self.kernel = ScaleKernel(self.kernel)

    def __init_kernel(self, kernel_str: str):
        match kernel_str:
            case "Matern5/2":
                return MaternKernel(nu=2.5)
            case "Matern3/2":
                return MaternKernel(nu=1.5)
            case "MaternLight":
                return MaternKernel(
                    nu=0.5
                )  # Smallest (most lightweight) approximation of the covariance matrix
            case "RBF":
                return RBFKernel()
            case _:
                raise ValueError("Incorrect Kernel initialization string")

    def forward(
        self,
        X: Any,
        distribution: Optional[callable] = MultivariateNormal,
        **kwargs,
    ) -> T.tensor:
        if type(X) == np.ndarray:
            X = T.tensor(X)
        mu = self.mean_fn.__call__(X)
        apprx_cov = self.kernel(X)
        try:
            return distribution(mean=mu, covariance_matrix=apprx_cov) + self.noise
        except:
            raise ValueError(
                "Incorrect distribution type to sample, distribution must take mu, sigma as args"
            )


class NumpyDragonGPR(DragonGPR):  # TODO: Test
    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        kernel: Optional[str] = "Matern5/2",
        scale_wrapper: Optional[bool] = True,
        alpha: Optional[float] = 1e-3,
        n_restarts_optimizer: Optional[int] = 25,
    ):
        super().__init__(train_inputs=train_x, train_targets=train_y, likelihood=None)
        self.kernel = self.__init_kernel(kernel)
        self.regressor = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        self.noise = alpha
        if scale_wrapper:
            self.kernel = ScaleKernel(self.kernel)

    def __init_kernel(self, kernel_str: str):
        match kernel_str:
            case "Matern5/2":
                kernel = Matern(nu=2.5, length_scale=0.0001)
                return kernel
            case "Matern3/2":
                kernel = Matern(nu=2.5, length_scale=0.0001)
                return kernel  # Smallest (most lightweight) approximation of the covariance matrix
            case "RBF":
                return RBF()
            case _:
                raise ValueError("Incorrect Kernel initialization string")

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self.regressor.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.regressor.predict(X)


class BanditosGPR(ExactGP):
    def __init__(
        self,
        train_x: T.tensor,
        train_y: T.tensor,
        likelihood: GaussianLikelihood,
        means: Optional[Any] = ZeroMean(),
        kernel: Optional[str] = "Matern5/2",
        scale_wrapper: Optional[bool] = True,
        alpha: Optional[float] = 1e-3,
        log_warp_scale: Optional[float] = 1.5,
    ):
        super().__init__(
            train_inputs=train_x, train_targets=train_y, likelihood=likelihood
        )
        self.mean_fn = means
        self.log_warp_scale = log_warp_scale
        self.kernel = self.__init_kernel(kernel)
        self.noise = alpha
        if scale_wrapper:
            self.kernel = ScaleKernel(self.kernel)

    def __init_kernel(self, kernel_str: str):
        match kernel_str:
            case "Matern5/2":
                return MaternKernel(nu=2.5)
            case "Matern3/2":
                return MaternKernel(nu=1.5)
            case "MaternLight":
                return MaternKernel(
                    nu=0.5
                )  # Smallest (most lightweight) approximation of the covariance matrix
            case "RBF":
                return RBFKernel()
            case _:
                raise ValueError("Incorrect Kernel initialization string")

    def linear_scaling(self, val):
        med = T.median(val)
        filtered_val = T.ones_like(val)
        T.where(val > med, val, filtered_val)
        sq_diff = T.sum((filtered_val - med) ** 2)
        if sq_diff == 0.0:
            sq_diff = (val - med) ** 2
        norm = T.sqrt(sq_diff)  # calc norm
        if not norm == 0.0:
            ret = val / norm
        else:
            ret = val
        return ret

    def half_rank_warping(self, val):
        ret = T.zeros_like(val)
        ret.copy_(val)

        boundary = T.median(val)
        mu = T.mean(val)
        sig = T.std(val)
        if sig == 0:
            return val

        vec = []
        replace = []
        for idx, x in enumerate(val):  # get all values < median
            if x < boundary:
                vec.append(x.item())
                replace.append(idx)
        scaled_vec = -T.abs(
            (T.tensor(vec) - mu) / sig
        )  # scale to lower half of the normal dist
        for idx, x in zip(replace, scaled_vec):
            ret[idx] = x

        return ret

    def log_warping(self, val):
        val = (T.max(val) - val) / (T.max(val) - T.min(val))  # normalize D[0,1]
        log_interior = 1 + (val * (self.log_warp_scale - 1.0))
        val = 0.5 - T.log(log_interior)
        val = val / np.log(self.log_warp_scale)
        return val

    def warp_output(self, val):
        val_ = self.linear_scaling(val)
        val_ = self.half_rank_warping(val_)
        val_ = self.log_warping(val_)
        assert val.shape == val_.shape
        return val_

    def forward(
        self,
        X: Any,
        distribution: Optional[callable] = MultivariateNormal,
        **kwargs,
    ) -> T.tensor:
        if type(X) == np.ndarray:
            X = T.tensor(X, **kwargs)
            X = (T.max(X) - X) / (T.max(X) - T.min(X))  # input norm
            X = T.log(X)
        mu = self.mean_fn.__call__(X)
        apprx_cov = self.kernel(X)
        val = None
        try:
            val: MultivariateNormal = (
                distribution(mean=mu, covariance_matrix=apprx_cov, validate_args=True)
                + self.noise
            )
        except:
            raise ValueError(
                "Incorrect distribution type to sample, distribution must take mu, sigma as args"
            )
        base_samples = val.get_base_samples()
        posterior = GPyTorchPosterior(val).rsample_from_base_samples(
            sample_shape=base_samples.size(), base_samples=base_samples
        )
        # value = val.get_base_samples()
        return self.warp_output(posterior)
