import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data.dataset import DefaultDataset, GeoDataset
from utils.file import list_all_images


def _make_balanced_sampler(label_list):
    """
    Creates a `sampler.WeightedRandomSampler` to balance class frequencies in the dataset.

    This is useful when the dataset is imbalanced (some classes appear more often then others).
    The sampler assigns each sample a weight inversely proportional to its class frequency.

    Args:
        label_list (List[int] or np.ndarray): A list of class labels for each sample in the dataset.

    Returns:
        WeightedRandomSampler: A sampler object for use with PyTorch DataLoader.
    """

    label_list = np.asarray(label_list, dtype=np.int64)
    class_counts = np.bincount(label_list)  # Count occurrences of each class
    class_weights = 1.0 / class_counts
    weight_list = class_weights[
        label_list
    ]  # Assign weights to each sample based on its class
    # param `replacement` is set to True to allow sampling with replacement
    return WeightedRandomSampler(weight_list, len(weight_list), replacement=True)


def get_train_loader(
    train_path, image_size, batch_size, dataset, num_workers=4, **kwargs
):
    """
    Returns a DataLoader for the training dataset.

    The transform applied depends on the dataset type. It supports class balancing using a weighted sampler.

    Args:
        train_path (str): Path to the training dataset.
        image_size (int): Size to which images will be resized.
        batch_size (int): Number of samples per batch.
        dataset (str): Type of dataset ('CelebA' or 'CUB2011').
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader for the training dataset.
    """

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if dataset == "CelebA":
        transform = transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
    elif dataset == "CUB2011":
        transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 76 / 64)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
    else:
        assert False, f"Unsupported dataset: {dataset}"

    dataset = ImageFolder(root=train_path, transform=transform)
    sampler = _make_balanced_sampler(dataset.targets)

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )


def get_test_loader(
    test_path, image_size, batch_size, dataset=None, num_workers=4, **kwargs
):
    """
    Returns a DataLoader for the test or validation dataset.

    Applies basic resizing and normalization. Does not perform any data augmentation.

    Args:
        test_path (str): Path to the test dataset.
        image_size (int): Size to which images will be resized.
        batch_size (int): Number of samples per batch.
        dataset (str): None (not used in this function).
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader for the test dataset.
    """

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    dataset = ImageFolder(root=test_path, transform=transform)

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )


def get_eval_loader(
    eval_path, image_size, batch_size, dataset=None, num_workers=4, **kwargs
):
    """
    Returns a DataLoader for unlabeled image folders used for evaluation.

    Assumes that the path contains only images (not subfolders). Does not shuffle the data.

    Args:
        eval_path (str): Path to the evaluation dataset.
        image_size (int): Size to which images will be resized.
        batch_size (int): Number of samples per batch.
        dataset (str): None (not used in this function).
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader for the evaluation dataset.
    """

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    dataset = DefaultDataset(root_dir=eval_path, transform=transform)

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )


def get_selected_loader(
    selected_path, image_size, dataset=None, num_workers=0, **kwargs
):
    """
    Returns a DataLoader for a small selected set of images.

    This is commonly used for visualizing selected images. Assumes all images are in a flat folder,
    and enforces a batch size limit of 64.

    Args:
        selected_path (str): Path to the selected images.
        image_size (int): Size to which images will be resized.
        dataset (str): None (not used in this function).
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader for the selected images.
    """

    # Selected path should be an image folder without subfolders.
    batch_size = len(list_all_images(selected_path))
    assert (
        batch_size < 64
    ), f"Batch size {batch_size} is too large. Please reduce the number of images in the selected folder."
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    dataset = DefaultDataset(root_dir=selected_path, transform=transform)

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )


def get_geo_train_loader(
    metadata_path,
    dataset_path,
    batch_size,
    crop_size=224,
    dataset=None,
    num_workers=4,
    **kwargs,
):
    """
    Returns a DataLoader for the training dataset with geographical coordinates.

    The transform applied includes resizing, random cropping, and normalization.

    Args:
        metadata_path (str): Path to the CSV metadata file containing image paths and coordinates.
        dataset_path (str): Path to the root directory containing images.
        crop_size (int): Size to which images will be cropped.
        batch_size (int): Number of samples per batch.
        dataset (str): None (not used in this function).
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional arguments for DataLoader.
    """

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    dataset = GeoDataset(
        metadata_path, dataset_path, transform=transform, use_cache=True
    )

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )


def get_geo_test_loader(
    metadata_path,
    dataset_path,
    batch_size,
    image_size=256,
    crop_size=224,
    dataset=None,
    num_workers=4,
    **kwargs,
):
    """
    Returns a DataLoader for the test dataset with geographical coordinates.

    The transform applied includes resizing and normalization.

    Args:
        metadata_path (str): Path to the CSV metadata file containing image paths and coordinates.
        dataset_path (str): Path to the root directory containing images.
        image_size (int): Size to which images will be resized.
        crop_size (int): Size to which images will be cropped.
        batch_size (int): Number of samples per batch.
        dataset (str): None (not used in this function).
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional arguments for DataLoader.
    """

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    dataset = GeoDataset(
        metadata_path, dataset_path, transform=transform, use_cache=True
    )

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )
