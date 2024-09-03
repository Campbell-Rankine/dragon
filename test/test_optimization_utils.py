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

from test_utils.test_models import LeNet, LSTMTS, LSTMTSConfig, device

X = T.normal(
    mean=T.zeros(size=(3, 3)),
    std=1,
)
Y = T.tensor([1.4, 1.43, 12.2])


class TestOptUtils:
    def test_init_gpr(self):
        likelihood = GaussianLikelihood()
        try:
            myGPR = DragonGPR(X, Y, likelihood=likelihood)
            assert True
        except:
            assert False

    def test_init_gpr_invalid_type(self):
        likelihood = GaussianLikelihood()
        try:
            myGPR = DragonGPR(X, Y, likelihood=likelihood, kernel="RandomKernel")
            assert False
        except:
            assert True

    def test_bopt_init(self):
        # initialize hyperparameters (test fn: lr cannot be a float % 2 == 0)
        def lr_constraint(value: float) -> bool:
            return int(100 * value) % 2 == 0

        lr_param = Hyperparameter(
            name="lr",
            type_="numerical",
            x=0.01,
            constraints=[Constraint(lr_constraint)],
            range_=(1e-10, 1e-2),
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
            hyper_params=[lr_param],
        )

    def test_fail_on_invalid_hyperparam_value(self):
        # initialize hyperparameters (test fn: lr cannot be a float % 2 == 0)
        def lr_constraint(value: float) -> bool:
            return int(100 * value) % 2 == 0

        try:
            lr_param = Hyperparameter(
                name="lr",
                type_="numerical",
                x=0.0,
                constraints=[Constraint(lr_constraint)],
                range_=(1e-10, 1e-2),
            )
        except ValueError:
            return True
        return False

    def test_bopt_sample_from_bounds(self):
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
        )

        restart_best = bopt._sample_next_points(xi=0.05)
        # check predicted LR

    def test_bopt_sample_from_bounds_expected_improvement(self):
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

        def bad_model(x, lr):
            return 10 * x[0][0]

        # objective function
        def obj(model: callable, input_):
            return model(input_)

        bopt = BaseDragonBayesOpt(
            objective_function=obj,
            Y_init=T.tensor([1.0]),
            hyper_params=[lr_param, beta0_param, beta1_param],
            acquisition_function=expected_improvement,
        )

        restart_best = bopt._sample_next_points(xi=0.05)
        # check predicted LR
