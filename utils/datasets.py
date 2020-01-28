import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import cv2
import os
import numpy as np
from .transforms import get_default_transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import multiprocessing.dummy as mp

n_threads = 16
p = mp.Pool(n_threads)

from numba import njit, jit, prange



@jit
def edge_promoting(root, save):

    #from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
    #import warnings

    #warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    #warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    #warnings.simplefilter('ignore', category=NumbaWarning)

    img_size = (384, 384)
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    
    pbar = tqdm.tqdm(total=len(file_list))
    
    job_args = [(os.path.join(root, f), gauss, img_size, kernel, kernel_size, save, n) for n, f in enumerate(file_list)]

    for _ in p.imap_unordered(edge_job, job_args):       
        pbar.update(1)

@njit()
def fast_loop(gauss_img, pad_img, kernel_size, gauss, dilation):
    idx = np.where(dilation != 0)
    loops = int(np.sum(dilation != 0))
    #print(gauss_img, pad_img, kernel_size, gauss, dilation)
    for i in range(loops):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
    return gauss_img

@jit
def edge_job(args):
    output_size = 256, 256
    path, gauss, img_size, kernel, kernel_size, save, n = args
    rgb_img = cv2.imread(path)
    gray_img = cv2.imread(path, 0)
    if rgb_img is None:
        print(path, "Error!")
        return
    rgb_img = np.array(ImageOps.fit(Image.fromarray(rgb_img), img_size, Image.ANTIALIAS))
    pad_img = np.pad(rgb_img, ((3, 3), (3, 3), (0, 0)), mode='reflect')
    gray_img = np.array(ImageOps.fit(Image.fromarray(gray_img), img_size, Image.ANTIALIAS))
    edges = cv2.Canny(gray_img, 150, 500) #200, 500 is good but maybe too little blur is applied
    dilation = cv2.dilate(edges, kernel)

    _gauss_img = np.copy(rgb_img)
    gauss_img = fast_loop(_gauss_img, pad_img, kernel_size, gauss, dilation)
    # idx = np.where(dilation != 0)

    # for i in range(np.sum(dilation != 0)):
    #     gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
    #         pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
    #     gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
    #         pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
    #     gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
    #         pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

    rgb_img = cv2.resize(rgb_img, output_size, Image.ANTIALIAS)
    gauss_img = cv2.resize(gauss_img, output_size)
    comb_img = np.concatenate((rgb_img, gauss_img), axis=1)
    #cv2.imwrite(os.path.join(save, str(n) + '.png'), comb_img)
    cv2.imwrite(os.path.join(save, str(n) + '.jpg'), comb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


class ImageDataset(Dataset):
    """Image dataset.""" 

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.root_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.root_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.root_files[idx]))
        if self.transform:
            img = self.transform(img)

        return img

def get_dataloader(path="./datasets/real_images", size=256, bs=64, trfs=None, flip=.005):
    "If no transforms is specified use default transforms"

    if not trfs:
        trfs = get_default_transforms(size=size)
    dset = ImageDataset(path, transform=trfs)
    return DataLoader(dset, batch_size=bs, num_workers=4, drop_last=True, shuffle=True)
