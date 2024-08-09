import psutil as __p
import torch as T
from typing import Optional, Tuple, Any, List, Literal


def get_resource_usage() -> Tuple[float, float, float]:
    cpu_usage = __p.cpu_percent()
    mem_usage = __p.virtual_memory().percent
    reserved, total = T.cuda.mem_get_info()
    gpu_usage = round(100.0 * float(reserved / total), 2)
    return cpu_usage, mem_usage, gpu_usage
