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
import cv2 as cv
from skimage.util import random_noise

def generate_noisy(sgm, img_path):
    """ generate noisy pairs """

    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    if img.dtype == np.uint8:
        img = img / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    noisy_img = random_noise(img, mode = 'gaussian', var = (sgm/255.)**2)

    # img_size = img.shape
    # noisy_img = img + sgm / 255.0 * np.random.randn(*img_size)

    # return (W, H, C) RGB image in range (0, 1)
    return img.astype(np.float32), noisy_img.astype(np.float32)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # set this to False, if being exactly deterministic is in need.

