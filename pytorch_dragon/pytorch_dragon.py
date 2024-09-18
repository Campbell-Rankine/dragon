"""
AI / Deep Learning Utilities Library (pytorch) (Python >=3.10)
"""

# TODO: Check Environment (packages on INIT)
# TODO: Finish setup.py

# Once all of the modules are finished deploy 0.1.0. For 0.2.0 pull some models into the models file.

# Additional Features: Basically just a list of helper functions for the user or setup/init utilities.

import numpy as np
from numba import cuda


def sobel_filter(img: np.ndarray) -> np.ndarray:
    """
    CUDA Optimized Sobel Filtering algorithm. Pass Binary image
    """
    assert len(img.shape) == 2  # assert two dimensional
    from .tools.vision import _cuda_sobel

    return _cuda_sobel(img)
