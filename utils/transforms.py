import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as t
from collections import namedtuple
from typing import Tuple, List
import os
import mimetypes
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO Handle (later maybe) edge processing transformation.


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # self.root_filenames = os.listdir(root_dir)
        self.root_filenames = [os.path.join(dp, f) for dp, dn, filenames in os.walk(
            root_dir) for f in filenames if mimetypes.guess_type(f)[0].startswith("image") and os.path.getsize(os.path.join(dp, f)) > 1]
        self.transform = transform

    def __len__(self):
        return len(self.root_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.root_filenames[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img


def train_transform(resolution: int):
    return t.Compose([
        t.Resize(resolution),
        t.RandomCrop(resolution),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize(*get_normalization_values())
    ])


def test_transform(resolution: int):
    return t.Compose([
        t.Resize(resolution),
        t.CenterCrop(resolution),
        t.ToTensor(),
        t.Normalize(*get_normalization_values())
    ])


def get_photo_train_loader(resolution: int, batch_size: int):
    # dataset = ImageDataset("./datasets/train_photos_places",
    #                        transform=train_transform(resolution))

    dataset = ImageDataset(r"D:\C_GAN\datasets\train_photos_places",
                           transform=train_transform(resolution))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def get_photo_test_loader(resolution: int, batch_size: int):
    dataset = ImageDataset("./datasets/test_photos",
                           transform=test_transform(resolution))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def get_cartoon_train_loader(resolution: int, batch_size: int):
    # dataset = ImageDataset("./datasets/train_cartoon_background",
    #                        transform=train_transform(resolution))

    dataset = ImageDataset(r"D:\C_GAN\datasets\train_cartoon_background",
                           transform=train_transform(resolution))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def get_normalization_values() -> Tuple[List[int], List[int]]:
    """Get the ImageNet mean/std values"""
    return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_no_aug_transform():
    return t.Compose([
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


def get_pair_transforms(size=256, flip=.01):
    return t.Compose([
        t.Resize(size=size),
        t.RandomHorizontalFlip(p=flip),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
