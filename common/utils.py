import numpy as np
import os
import torch
import random

def set_seed(seed: int = 42):
    """保证训练过程可以复现.
    参数:
        seed (int): 随机种子.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False