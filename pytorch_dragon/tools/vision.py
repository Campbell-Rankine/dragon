"""
Vision model extension modules. Apply as layers on nn.Module classes
"""

import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from numba import cuda
import numpy as np
import math
from typing import Optional
import cv2 as cv


# Feature Extraction
@cuda.jit  # Tell numba jit compiler it needs to map cpu functions to cuda functions
def sobel_filter(input_image, output_image):
    # Apply sobel filter inplace on the output image.
    x, y = cuda.grid(2)

    if x < input_image.shape[0] - 2 and y < input_image.shape[1] - 2:
        # get Gx
        Gx = (
            input_image[x, y]
            - input_image[x + 2, y]
            + 2 * input_image[x, y + 1]
            - 2 * input_image[x + 2, y + 1]
            + input_image[x, y + 2]
            - input_image[x + 2, y + 2]
        )
        # get Gy
        Gy = (
            input_image[x, y]
            - input_image[x, y + 2]
            + 2 * input_image[x + 1, y]
            - 2 * input_image[x + 1, y + 2]
            + input_image[x + 2, y]
            - input_image[x + 2, y + 2]
        )

        # in place op
        output_image[x + 1, y + 1] = math.sqrt(Gx**2 + Gy**2)


def cuda_sobel(
    np_image: np.ndarray,
    kernel_size: Optional[int] = 16,
    variance_scaler: Optional[int] = 4,
):
    np_image = cv.GaussianBlur(
        np_image,
        (kernel_size, kernel_size),
        np.std(np_image) / variance_scaler,  # Gaussian filter for smoother gradients
    )
    # alloc
    cuda_im = cuda.to_device(np_image)
    output_image_ = np.zeros_like(np_image)
    threads_per_block = (kernel_size, kernel_size)

    # calculate dims
    blockspergrid_x = (
        np_image.shape[0] + threads_per_block[0] - 1
    ) // threads_per_block[0]
    blockspergrid_y = (
        np_image.shape[1] + threads_per_block[1] - 1
    ) // threads_per_block[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # apply sobel filter
    sobel_filter[blockspergrid, threads_per_block](cuda_im, output_image_)
    return output_image_


# Neural Network Modules
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
