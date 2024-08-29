import torch as T
import numpy as np
from gpytorch.models import ExactGP
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
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
            X = T.tensor(X, **kwargs)
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
