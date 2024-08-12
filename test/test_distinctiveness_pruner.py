import torch as T
import torch.nn
import pytest

from tools.pruning import DistinctivenessPruning
from test_utils.test_models import LeNet, LSTMTS, LSTMTSConfig, device


class TestDistinctivenessPrune:
    def test_get_modules(self):
        # test module retrieval
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)

        pruner = DistinctivenessPruning(model, tolerance=(30, 150), device=device)
        m1, m2 = pruner.current_modules()
        assert m1 == "lstm" and m2 == "fc2"

    def test_angle_calc_ts(self):
        # test angle calc
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)

        pruner = DistinctivenessPruning(model, tolerance=(30, 150), device=device)
        m1, m2 = pruner.current_modules()
        assert False  # TODO: Finish test
