import torch as T
import torch.nn as nn
import json

from utils.metrics import gradient_norm
from utils.system import get_resource_usage

metrics = [get_resource_usage, gradient_norm]
