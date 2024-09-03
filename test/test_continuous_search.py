import torch as T
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F
from botorch.acquisition import ProbabilityOfImprovement, ExpectedImprovement
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood

from search.acquisition.functions import probability_improvement, expected_improvement
from search.models.bopt_models import DragonGPR
from search.continuous_search import BaseDragonBayesOpt
from search.hyperparam import Hyperparameter, Constraint, RangeConstraint
from typing import Optional

from test_utils.test_models import LeNet, LSTMTS, LSTMTSConfig, device
import pytest

# TODO: Write and test bayes opt call / fit / predict functions


class TestBayesOpt:
    def test_bopt_iteration_stop(self):
        # initialize hyperparameters (test fn: lr cannot be a float % 2 == 0)
        T.manual_seed(0)
        np.random.seed(0)

        def lr_constraint(value: float) -> bool:
            return int(100 * value) % 2 == 0

        lr_param = Hyperparameter(
            name="lr",
            type_="numerical",
            x=0.01,
            constraints=[Constraint(lr_constraint)],
            range_=(1e-10, 1e-2),
        )

        beta0_param = Hyperparameter(
            name="beta0",
            type_="numerical",
            x=0.91,
            range_=(0.9, 0.95),
        )

        beta1_param = Hyperparameter(
            name="beta1",
            type_="numerical",
            x=0.97,
            range_=(0.95, 0.999999),
        )
        print("Finished lr param init")

        def bad_model(x, lr):
            return 10 * x[0][0]

        # objective function
        def obj(model: callable, input_):
            return model(input_)

        bopt = BaseDragonBayesOpt(
            objective_function=obj,
            Y_init=T.tensor([1.0]),
            hyper_params=[lr_param, beta0_param, beta1_param],
            iters=0,
        )

        result = bopt(model=bad_model, batch_X=T.tensor([1]), batch_Y=T.tensor([0.0]))
        assert result["Y"] == T.tensor([1.0])
        assert bopt.stop

    def test_two_iters_lstm(self):
        # seed
        T.manual_seed(0)
        np.random.seed(0)

        # model init
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)

        lr_param = Hyperparameter(
            name="lr", type_="numerical", x=0.01, range_=(1e-10, 1e-2)
        )

        beta0_param = Hyperparameter(
            name="beta0", type_="numerical", x=0.91, range_=(0.9, 0.95)
        )

        beta1_param = Hyperparameter(
            name="beta1", type_="numerical", x=0.97, range_=(0.95, 0.999999)
        )

        x_train = T.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.05, 0.25, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.1, 0.3, 0.3, 0.6, 0.6],
            ],
        )
        y_train = T.tensor([[1.0], [1.0], [0.0]])

        def obj(model, X, y, hprev, cprev, prediction_name: Optional[str] = "call"):
            count = 0.0
            for i in range(X.shape[0]):
                x = X[i]
                (y, _) = model(X, hprev=hprev, cprev=cprev)
                count += T.sum(y)
            return count

        bopt = BaseDragonBayesOpt(
            objective_function=obj,
            Y_init=T.tensor([0.5]),
            hyper_params=[lr_param, beta0_param, beta1_param],
            iters=50,
        )

        result1 = bopt(
            model,
            x_train,
            y_train,
            hprev=T.zeros(1, 1, 24),
            cprev=T.zeros(1, 1, 24),
        )

        result2 = bopt(
            model,
            x_train,
            y_train,
            hprev=T.zeros(1, 1, 24),
            cprev=T.zeros(1, 1, 24),
        )

        # check improvement + different results
        assert not (-1 * result1["value"] == -1 * result2["value"])

        # check all hyperparams have their values set.
        returned_vals = result2["x"]
        for i in range(len(result2["x"])):
            assert returned_vals[i] == bopt.params[i].value

    def test_banditos(self):
        # seed
        T.manual_seed(0)
        np.random.seed(0)

        # model init
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)

        lr_param = Hyperparameter(
            name="lr", type_="numerical", x=0.01, range_=(1e-10, 1e-2), log_range=True
        )

        beta0_param = Hyperparameter(
            name="beta0", type_="numerical", x=0.91, range_=(0.9, 0.95), log_range=True
        )

        beta1_param = Hyperparameter(
            name="beta1",
            type_="numerical",
            x=0.97,
            range_=(0.95, 0.999999),
            log_range=True,
        )

        x_train = T.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.05, 0.25, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.1, 0.3, 0.3, 0.6, 0.6],
            ],
        )
        y_train = T.tensor([[1.0], [1.0], [0.0]])

        def obj(model, X, y, hprev, cprev, prediction_name: Optional[str] = "call"):
            count = 0.0
            for i in range(X.shape[0]):
                x = X[i]
                (y, _) = model(X, hprev=hprev, cprev=cprev)
                count += T.sum(y)
            return count

        bopt = BaseDragonBayesOpt(
            objective_function=obj,
            Y_init=T.tensor([0.5]),
            hyper_params=[lr_param, beta0_param, beta1_param],
            iters=50,
            regressor_type="banditos",
        )

        result1 = bopt(
            model,
            x_train,
            y_train,
            hprev=T.zeros(1, 1, 24),
            cprev=T.zeros(1, 1, 24),
        )

        result2 = bopt(
            model,
            x_train,
            y_train,
            hprev=T.zeros(1, 1, 24),
            cprev=T.zeros(1, 1, 24),
        )

        # check improvement + different results
        assert not (-1 * result1["value"] == -1 * result2["value"])

        # check all hyperparams have their values set.
