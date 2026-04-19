import numpy as np
import torch

def get_device():
    if not torch.cuda.is_available():
        return "cpu"
    else:
        nb_gpu = torch.cuda.device_count()
        return np.arange(nb_gpu).tolist()
    return "cpu"