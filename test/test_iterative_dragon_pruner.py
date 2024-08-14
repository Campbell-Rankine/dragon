"""
Testing functions related to the iterative base class pruner
"""

import torch as T
import torch.nn as nn
import pytest

from tools.pruning import IterativeDragonPruner
from test_utils.test_models import LeNet, LSTMTS, LSTMTSConfig, device

# test data

# ground truth LeNet module pair list
module_list = ["conv1", "conv2", "fc1", "fc2", "fc3"]
module_list2 = ["lstm", "fc2"]


class TestDragonPruner:
    # test function for next module
    def test_get_modules(self):
        model = LeNet(0.0)
        pruner = IterativeDragonPruner(model)
        counter = 0
        while pruner.STOP_FLAG == False:
            counter += 1
            pruner._next_module()

            # check pair against ground truth
            m = pruner.current_modules()

            m1, _ = m["curr"]

            if not pruner.STOP_FLAG:
                ground_truth_1 = module_list[counter]

                assert m1 == ground_truth_1

            if counter > 100:
                assert False  # break from test

        assert (
            counter <= 5 and pruner.STOP_FLAG == True
        )  # make sure stop flag and counter work

    # test apply for IterativeDragonPruner
    def test_apply_func_lenet(self):
        model = LeNet(0.0)
        pruner = IterativeDragonPruner(model)

        # define pruning function
        def prune_func(wi: T.tensor, wj: T.tensor, model: nn.Module):
            return T.zeros_like(wi)

        with T.no_grad():
            result = pruner.apply(prune_func, model, over_next_layer=True)

        for idx, (k, v) in enumerate(result.items()):
            assert k == module_list[idx]
            for value in v:
                k2 = list(value.keys())[0]
                assert T.sum(value[k2]) == 0
                assert T.sum(model.state_dict()[f"{k}.{k2}"]) == 0

    def test_apply_func_lstm(self):
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)
        pruner = IterativeDragonPruner(model)

        # define pruning function
        def prune_func(wi: T.tensor, wj: T.tensor, model: nn.Module):
            return T.zeros_like(wi)

        with T.no_grad():
            result = pruner.apply(prune_func, model, over_next_layer=True)

        for idx, (k, v) in enumerate(result.items()):
            assert k == module_list2[idx]
            for value in v:
                k2 = list(value.keys())[0]
                assert T.sum(value[k2]) == 0
                assert T.sum(model.state_dict()[f"{k}.{k2}"]) == 0
