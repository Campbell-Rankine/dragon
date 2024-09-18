"""
Native pytorch implementations of BayesOpt acquisition functions
"""

import torch as T
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F
from gpytorch.models import ExactGP
from typing import Optional, Any, List, Tuple, Dict


def probability_improvement(
    X: np.ndarray,
    X_sample: T.tensor,
    regressor: ExactGP,
    xi: Optional[float] = 0.1,
    *tensor_kwargs,
    **regressor_kwargs,
) -> T.tensor:
    """
    Probability Improvement Acquisition function
    Compute probability of improvement at X using Sample points X_sample

    Args:
        X (T.tensor): Point to compute PI
        X_Sample (T.tensor): Array of sampled locations
        gpr (gpytorch.models.ExactGP): Gaussian Process Regressor
        xi (float): Trade-off Param(From PI definition)
        tensor_kwargs (Tuple[Any]): Tensor kwargs
        regressor_kwargs (Dict[str, Any]): Regressor.forward kwargs
    Returns:
        PI(X, X_hat) : T.tensor
    """
    X = T.tensor(X, *tensor_kwargs)
    # check regressor class object
    has_forward = getattr(
        regressor, "forward", None
    )  # pass default value None to stop getattr from raising an exception
    assert callable(has_forward)

    # calculate stddev of X
    stdX = T.std(X)
    if stdX == 0:
        return T.tensor(0.0, *tensor_kwargs)

    # Get sampled Z-scores
    pred = regressor.forward(X_sample.type(T.float), **regressor_kwargs)
    Z = (T.mean(X, dim=1) - T.mean(pred.sample(X.shape)) - xi) / stdX
    result = Normal(0, 1).cdf(Z)
    return result


def expected_improvement(
    X: np.ndarray,
    X_Sample: T.tensor,
    regressor: ExactGP,
    xi=0.01,
    *tensor_args,
    **regressor_kwargs,
):
    """
    Expected Improvement Acquisition improvement
    Instead of probability of improvement compute the expected improvement

    Args: Same as above
    """
    X = T.tensor(X, *tensor_args)
    # check regressor class object
    has_forward = getattr(
        regressor, "forward", None
    )  # pass default value None to stop getattr from raising an exception
    assert callable(has_forward)
    # calculate stddev of X
    stdX = T.std(X)
    if stdX == 0:
        return T.tensor(0.0, *tensor_args)

    # Get sampled Z-scores
    pred = regressor.forward(X_Sample.type(T.float), **regressor_kwargs)
    try:
        Zi = T.mean(X, dim=1) - T.mean(pred.sample(X.shape)) - xi
    except:
        Zi = T.mean(X, dim=1) - T.mean(pred) - xi
    Z = (Zi) / stdX
    return Zi * Normal(0, 1).cdf(Z) + stdX * Normal(0, 1).log_prob(Z).exp()
