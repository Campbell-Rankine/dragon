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

    def test_merge_neurons_lenet(self):
        T.manual_seed(0)
        model = LeNet(0.0)
        pruner = DistinctivenessPruning(tolerance=(30, 150), model=model, device=device)
        m = pruner.current_modules()
        m1, module1 = m["curr"]

        # retrieve weight vector from modules
        param1 = next(module1.parameters())
        wv1 = param1[0]
        wv2 = param1[1]
        result1, result2 = pruner.merge_neurons(wv1, wv2)
        assert T.sum(result1).item() - -0.4820 <= 0.0001
        assert T.sum(result2).item() == 0

    def test_prune_parameter_lenet(self):
        T.manual_seed(0)
        model = LeNet(0.0)

        old_weights = model.__getattr__("fc2").weight
        first_sum: float = T.sum(old_weights)

        pruner = DistinctivenessPruning(
            tolerance=(79, 99), model=model, device=device
        )  # Weights mostly orthogonal, average doesn't really change much
        model_, result = pruner(
            "fc2.weight", model, over_next_layer=True, return_result_dict=True
        )

        weights1 = model_.__getattr__("fc2").weight
        weights2 = old_weights
        assert weights1.shape == weights2.shape

        print(T.sum(weights1), first_sum)
        assert not (T.sum(weights1) == first_sum)

    def test_prune_parameter_lstm(self):
        # weights won't be pruned. Assert weights equal
        T.manual_seed(0)
        model_config = LSTMTSConfig()
        model = LSTMTS(model_config)

        old_weights = model.__getattr__("fc2").weight
        first_sum: float = T.sum(old_weights)
        print(first_sum)

        pruner = DistinctivenessPruning(
            tolerance=(79, 99), model=model, device=device
        )  # Weights mostly orthogonal, average doesn't really change much
        model_, result = pruner(
            "fc2.weight", model, over_next_layer=True, return_result_dict=True
        )
        weights1 = model_.__getattr__("fc2").weight
        weights2 = old_weights
        assert weights1.shape == weights2.shape
        assert T.sum(weights1) == first_sum

    def test_invalid_parameter_lenet(self):
        T.manual_seed(0)
        model = LeNet(0.0)

        old_weights = model.__getattr__("conv1").weight
        first_sum: float = T.sum(old_weights)

        pruner = DistinctivenessPruning(tolerance=(79, 99), model=model, device=device)
        try:  # Weights mostly orthogonal, average doesn't really change much
            model_, result = pruner(
                "conv1.weight", model, over_next_layer=True, return_result_dict=True
            )
            assert False

        except ValueError:
            assert True
