import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import torch.autograd as autograd

from utils import *

def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i) 
    np.random.seed(i)
    random.seed(i)
    
set_global_seeds(1)