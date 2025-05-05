import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.file import (
    exist_cache,
    list_all_images,
    load_cache,
    safe_filename,
    save_cache,
)


class DefaultDataset(Dataset):
    """
    A custom PyTorch dataset for loading unlabeled images from a directory.

    This dataset assumes that all images are stored under `root_dir` recursively.
    It supports optional image transformation (e.g., resizing, normalization).

    Attributes:
        samples (List[str]): List of file paths to images.
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset by recursively listing all images in the given root directory.

        Args:
            root_dir (str): Path to the root directory containing images.
            transform (callable): A optional function/transform to apply to each image.
        """

        self.samples = list_all_images(root_dir)
        self.samples.sort()
        self.transform = transform

    def load_image(self, image_path):
        """
        Loads an image from the given path and applies the specified transformation if provided.

        Args:
            image_path (str): Path to the image file.

        Returns:
            PIL.Image or Tensor: Loaded image, transformed if a transform is specified.
        """

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        """
        Fetches the image at the specified index from the dataset.

        Args:
            index (int): Index of the image to fetch.

        Returns:
            PIL.Image or Tensor: The image at the specified index, transformed if a transform is specified.
        """
        return self.load_image(self.samples[index])

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.samples)


class GeoDataset(Dataset):
    """
    A custom PyTorch dataset that loads RGB images and their corresponding geographical coordinates
    (latitude, longitude) from a CSV metadata file.

    This dataset is useful for geospatial learning tasks such as geolocation prediction, Earth observation,
    or geo-aware image image retrieval. Caching is supported via pickle to speed up repeated loading.

    Attributes:
        dataset_path (str): Path to the root directory containing images.
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        images (List[str]): List of file paths to images.
        coordinates (List[Tuple[float, float]]): List of tuples containing latitude and longitude for each image.
    """

    def __init__(self, metadata_path, dataset_path, transform=None, use_cache=False):
        """
        Initializes the GeoDataset by parsing the metadata and preparing image/coordinate lists.

        Args:
            metadata_path (str): Path to the CSV file with columns "IMG_ID", "LAT", and "LON".
            dataset_path (str): Root directory where image files are stored.
            transform (callable): A optional function/transform to apply to each image.
            use_cache (bool): If True, uses cached data if available.

        Returns:
            None
        """

        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.images, self.coordinates = self.load_geodata(metadata_path, use_cache)

    def load_geodata(self, metadata_path, use_cache=False):
        """
        Parses the metadata CSV file and collects valid image paths and their coordinates.
        Utilizes a cache to avoid redundant processing on subsequent runs.

        Args:
            metadata_path (str): Path to the CSV file.
            use_cache (bool): If True, uses cached data if available.

        Returns:
            Tuple[List[str], List[Tuple[float, float]]]: A tuple containing a list of image paths
                and a list of coordinates.
        """

        metafile_name = os.path.basename(metadata_path).split(".")[0]
        cachefile_name = safe_filename(metafile_name + "_geodata_cache.pkl")
        if use_cache and exist_cache(cachefile_name):
            cache_data = load_cache(cachefile_name)
            if isinstance(cache_data, tuple) and len(cache_data) == 2:
                print(f"Loading {cachefile_name} from cache...")
                return cache_data
            else:
                print(
                    f"[Cache Warning] Invalid cache data format in {cachefile_name}. Reprocessing metadata."
                )

        try:
            metadata_info = pd.read_csv(metadata_path)
        except Exception as e:
            raise IOError(f"[IOError] Failed to load metadata file: {e}")
        images, coordinates = [], []

        for _, row in tqdm(
            metadata_info.iterrows(),
            desc="Loading image paths and coordinates",
            total=len(metadata_info),
        ):
            image_path = os.path.join(self.dataset_path, row["IMG_ID"])
            if os.path.exists(image_path):
                images.append(image_path)
                latitude = float(row["LAT"])
                longitude = float(row["LON"])
                coordinates.append((latitude, longitude))
            else:
                print(f"[Warning] Image {image_path} does not exist. Skipping.")

        if use_cache:
            save_cache((images, coordinates), cachefile_name)
            print(f"Saved {cachefile_name} to cache.")

        return images, coordinates

    def __getitem__(self, index):
        """
        Fetches the image and its corresponding geographical coordinates at the specified index.

        Args:
            index (int): Index of the image to fetch.

        Returns:
            Tuple[PIL.Image, torch.Tensor]: A tuple containing the loaded image and its coordinates.
                The image is transformed if a transform is specified.
        """

        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        coordinate = torch.tensor(self.coordinates[index], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, coordinate

    def __len__(self):
        """
        Returns the total number of valid samples in the dataset.

        Returns:
            int: Number of valid samples in the dataset.
        """
        return len(self.images)
