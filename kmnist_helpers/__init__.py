import torch
import random
import numpy as np

def set_seed(seed):
    """ Use this to set ALL the random seeds to a fixed value and take out any 
        randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #uses inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled   = False

    return True

seed = 42
device = 'cpu'

if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")