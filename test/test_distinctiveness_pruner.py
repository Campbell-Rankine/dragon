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
        m = pruner.current_modules()
        m1, _ = m["curr"]
        m2, _ = m["next"]
        assert m1 == "lstm" and m2 == "fc2"

    def test_angle_calc_ts(self):
        # test angle calc (result comparison was calculated by hand)
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)

        # get modules
        pruner = DistinctivenessPruning(tolerance=(30, 150), model=model, device=device)
        m = pruner.current_modules()
        m1, module1 = m["curr"]

        # retrieve weight vector from modules
        param1 = next(module1.parameters())
        wv1 = param1[0]
        wv2 = param1[1]

        result = pruner.get_angle(wv1, wv2)
        assert int(result) == 72  # some wiggle room to allow for fp
