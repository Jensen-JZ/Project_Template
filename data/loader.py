from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder # Use official ImageFolder

from data.dataset import DefaultDataset
# Removed: from utils.file import list_all_images - not needed after removing get_selected_loader
# Removed: import numpy as np - not needed after removing _make_balanced_sampler
# Removed: from torch.utils.data.sampler import WeightedRandomSampler - not needed


def get_train_loader(train_path, input_shape, batch_size, num_workers=4, **kwargs):
    # Generic transform
    transform = transforms.Compose([
        transforms.Resize([input_shape[0], input_shape[1]]), # Use input_shape
        transforms.RandomHorizontalFlip(), # Common augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Generic normalization
    ])

    dataset = ImageFolder(root=train_path, transform=transform)
    # Removed sampler logic, using default shuffle=True

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True, # Standard shuffling
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_test_loader(test_path, input_shape, batch_size, num_workers=4, **kwargs):
    # Generic transform
    transform = transforms.Compose([
        transforms.Resize([input_shape[0], input_shape[1]]), # Use input_shape
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Generic normalization
    ])
    dataset = ImageFolder(root=test_path, transform=transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False, # Typically False for test/eval
                           num_workers=num_workers,
                           pin_memory=True)


def get_eval_loader(path, input_shape, batch_size, num_workers=4, **kwargs):
    # Path should be an image folder without sub folders.
    # Generic transform
    transform = transforms.Compose([
        transforms.Resize([input_shape[0], input_shape[1]]), # Use input_shape
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Generic normalization
    ])
    dataset = DefaultDataset(root=path, transform=transform) # DefaultDataset for flat folder structure

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=False)

# Removed get_selected_loader as selected_path was removed from config
# Removed _make_balanced_sampler as it's not universally needed
