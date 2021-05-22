import torch
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms
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


class NAC_singleImg(Dataset):
    def __init__(self, imgy, sgm2, mode='train'):
        self.imgy = imgy
        self.sgm2 = sgm2
        self.mode = mode

    def __len__(self):
        # every epoch contains 16 different realizations (randomly set)
        return 4

    def __getitem__(self, idx):
        imgy = self.imgy

        # imgz = random_noise(imgy, mode = 'gaussian', var = (self.sgm2/255.)**2).astype(np.float32)

        img_size = imgy.shape
        imgz = imgy + self.sgm2 / 255.0 * np.random.randn(*img_size).astype(np.float32)


        # crop_size = (256, 256)
        # if crop_size is not None:
        #     h, w = imgy.shape[:2]
        #     new_h, new_w = crop_size

        #     top = np.random.randint(0, h - new_h)
        #     left = np.random.randint(0, w - new_w)

        #     id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        #     id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        #     imgy = imgy[id_y, id_x]
        #     imgz = imgz[id_y, id_x]

        # hflip = np.random.random() < 0.5
        # if hflip:
        #     imgy = np.fliplr(imgy)
        #     imgz = np.fliplr(imgz)

        # vflip = np.random.random() < 0.5
        # if vflip:
        #     imgy = np.flipud(imgy)
        #     imgz = np.flipud(imgz)

        # rotate= np.random.random() < 0.5
        # if rotate:
        #     degree = np.random.randint(-10, 10)
        #     imgy = transforms.functional.rotate(imgy, degree)
        #     imgz = transforms.functional.rotate(imgz, degree)


        imgy = torch.from_numpy(imgy.transpose(2, 0, 1).copy())
        imgz = torch.from_numpy(imgz.transpose(2, 0, 1).copy())

        return imgy, imgz


