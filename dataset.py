"""
dataset.py — CIFAR-10 / CIFAR-100 data loaders with automatic download.

The paper evaluates on CIFAR-10 (default) and CIFAR-100.
Images are loaded as [0,1] tensors (no channel-wise normalization in the
transform so that adversarial perturbations operate in pixel space).
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloaders(batch_size=128, data_dir='./data', num_workers=2):
    """Return train and test DataLoaders for CIFAR-10."""
    os.makedirs(data_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar100_dataloaders(batch_size=128, data_dir='./data', num_workers=2):
    """Return train and test DataLoaders for CIFAR-100."""
    os.makedirs(data_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader
