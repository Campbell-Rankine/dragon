import pytest
from backprop.grad_accumulator import GradAccumulator
import torch as T
import logging
import numpy as np

initial_data = T.tensor(
    data=[1.0, 1.1, 1.21],
    requires_grad=False,
)

test_data = T.tensor(
    data=[1.51, 1.6, 1.1, 1.3, 0.9, 0.8, 0.8, 1.1, 0.4, 0.5],
    requires_grad=False,
)

test_data_2 = T.tensor(
    data=[1.0, 1.1, 1.21, 1.51, 1.6, 1.1, 1.3, 0.9, 0.8, 0.8, 1.1, 0.4, 0.5],
    requires_grad=False,
)


class Test_Package:
    # test accumulator init
    def test_accum_init(self):
        try:
            accum = GradAccumulator(K=5, start_data=initial_data, requires_grad=False)
        except:
            assert False

        assert True

    # test accumulator window length bound
    def test_accum_length_bound(self):
        accum = GradAccumulator(K=5, start_data=initial_data, requires_grad=False)
        for i in range(10):
            item = test_data[i]
            output = accum(item)
            if len(accum.window.data) > accum.K:
                assert False
        assert True

    # test accumulator window length bound
    def test_accum_iter_accuracy(self):
        accum = GradAccumulator(K=3, start_data=initial_data, requires_grad=False)
        first_3_vals = [1.2733, 1.4400, 1.4033]  # ground truth values for test
        correct = 0
        for i in range(3):
            item = test_data[i]
            output = accum(item)
            if output.item() - first_3_vals[i] <= 0.0001:
                correct += 1
        if correct == 3:
            assert True
        else:
            assert False

    def test_accum_in_training_run(self):
        accum = GradAccumulator(K=3, requires_grad=False)
        ground_truth = [
            1.0,
            1.05,
            1.1033,
            1.2733,
            1.4400,
            1.4033,
            1.3333,
            1.1000,
            1.0000,
            0.8333,
            0.9000,
            0.7667,
            0.6667,
        ]  # all have been hand calculated
        for i in range(len(test_data_2)):
            item = test_data_2[i]
            output: T.tensor = accum(item)
            assert (output.item() - ground_truth[i]) <= 0.0001

    def test_accum_not_modify_tensor_params(self):
        accum = GradAccumulator(K=3, start_data=initial_data, requires_grad=False)
        for i in range(len(test_data_2)):
            item = test_data_2[i]
            output: T.tensor = accum(item)
            assert item.requires_grad == output.requires_grad
