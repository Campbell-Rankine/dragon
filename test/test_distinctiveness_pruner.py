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

        pruner = DistinctivenessPruning(tolerance=(30, 150), model=model, device=device)
        m1, m2 = pruner.current_modules()
        assert list(m1.keys())[0] == "lstm" and list(m2.keys())[0] == "fc2"

    def test_angle_calc_ts(self):
        # test angle calc
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)

        # get modules
        pruner = DistinctivenessPruning(tolerance=(30, 150), model=model, device=device)
        m1, _ = pruner.current_modules()

        # retrieve weight vector from modules
        [(_, module1)] = m1.items()
        param1 = next(module1.parameters())
        wv1 = param1[0]
        wv2 = param1[1]

        result = pruner.get_angle(wv1, wv2)
        assert int(result) == 72  # some wiggle room to allow for fp
