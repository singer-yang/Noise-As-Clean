'''
Author: your name
Date: 2021-04-25 17:03:03
LastEditTime: 2021-04-27 23:17:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /raw_denoising/utils/utils.py
'''

import torch
import numpy as np
import random 



def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # set this to False, if being exactly deterministic is in need.

