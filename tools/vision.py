"""
Vision model extension modules. Apply as layers on nn.Module classes
"""

import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class EqualizedLR_Conv2d(nn.Module):
    """
    Equalized LR Convolutional 2d cell. Used to prevent exploding gradients
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = np.sqrt(2 / (in_ch * kernel_size[0] * kernel_size[1]))

        self.weight = Parameter(T.Tensor(out_ch, in_ch, *kernel_size))
        self.bias = Parameter(T.Tensor(out_ch))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(
            x, self.weight * self.scale, self.bias, self.stride, self.padding
        )


class Pixel_norm(nn.Module):
    """
    Pixel wise normalization
    """

    def __init__(self):
        super().__init__()

    def forward(self, a):
        b = a / T.sqrt(T.sum(a**2, dim=1, keepdim=True) + 10e-8)
        return b


class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = T.std(x, dim=0)
        mean = T.mean(std)
        return T.cat((x, mean.repeat(size)), dim=1)


class fromRGB(nn.Module):
    """
    Learned conversion of a 3 channel image to a 1 channel image
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.cvt = EqualizedLR_Conv2d(in_c, out_c, (1, 1), stride=(1, 1))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.cvt(x)
        return self.relu(x)


class toRGB(nn.Module):
    """
    Learned conversion of a 1 channel image to a 3 channel image
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.cvt = EqualizedLR_Conv2d(in_c, out_c, (1, 1), stride=(1, 1))

    def forward(self, x):
        return self.cvt(x)
