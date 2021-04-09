import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils


def get_default_transforms(size=256):
    return transforms.Compose([
        transforms.Resize(size=size),
        transforms.CenterCrop(size=size),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_no_aug_transform(size=256):
    return transforms.Compose([
        # transforms.Resize(size=size),
        # transforms.CenterCrop(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_pair_transforms(size=256, flip=.01):
    return transforms.Compose([
        # transforms.Resize(size=size),
        # transforms.RandomHorizontalFlip(p=flip),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
