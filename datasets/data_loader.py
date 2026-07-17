import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import random_split
from configs import DATASET_PATH

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def load_train_val_datasets():
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=val_transform)

    generator = torch.Generator().manual_seed(42)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    return train_dataset, val_dataset

def load_test_dataset():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_dataset = CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)

    return test_dataset

def create_data_loaders(batch_size=64):
    train_dataset, val_dataset = load_train_val_datasets()
    test_dataset = load_test_dataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader