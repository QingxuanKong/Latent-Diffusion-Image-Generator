from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np


def build_val_loader(
    dataset_name,
    val_data_dir,
    data_dir,
    image_size,
    batch_size,
    num_workers,
    subset_ratio=1.0,
    distributed=False,
    world_size=1,
    rank=0
):
    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Load dataset
    if dataset_name == "imagenet100":
        val_dataset = datasets.ImageFolder(val_data_dir, transform=transform_val)
    elif dataset_name == "cifar10":
        val_dataset = datasets.CIFAR10(
            data_dir, train=False, transform=transform_val, download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Optional subset
    subset_size = int(len(val_dataset) * subset_ratio)
    indices = np.random.choice(len(val_dataset), subset_size, replace=False)
    val_dataset = Subset(val_dataset, indices)

    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    return val_loader
