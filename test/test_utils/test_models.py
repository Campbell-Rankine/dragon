import torch as T
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

device = T.device("cuda" if T.cuda.is_available() else "cpu")


# Image model
class LeNet(nn.Module):
    def __init__(self, seed: Optional[float] = 0.0):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # seed
        T.manual_seed(seed)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# time series model
class LSTMTS(nn.Module):
    def __init__(self, config, seed: Optional[float] = 0.0):
        super().__init__()
        self.hidden_layer_size = config.hidden_layer_size

        self.lstm = nn.LSTM(config.input_size, config.hidden_layer_size, 1)

        self.fc2 = nn.Linear(config.hidden_layer_size, config.output_size)

        self.hidden_cell = (
            T.randn(1, 1, self.hidden_layer_size),
            T.randn(1, 1, self.hidden_layer_size),
        )
        self._initialization_constraints(config)
        # seed
        self.seed = 0
        T.manual_seed(seed)

    def forward(self, x, hprev, cprev):
        lstm_out, hc = self.lstm(x.view(len(x), 1, -1), (hprev, cprev))
        out = self.fc2(lstm_out.view(len(x), -1))
        out = F.sigmoid(out)
        # We find unequal lengths in the data so you have to pick the correct final prediction value.
        return out, hc

    def _initialization_constraints(self, config) -> None:
        # constraints
        assert not config is None
        assert not config.hidden_layer_size is None
        assert not config.input_size is None
        assert not config.output_size is None


# config class for LSTM
class LSTMTSConfig:
    def __init__(
        self,
        input_size: Optional[int] = 6,
        hidden_layer_size: Optional[int] = 24,
        output_size: Optional[int] = 1,
    ):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
