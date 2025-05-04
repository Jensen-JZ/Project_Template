import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.file import (
    exist_cache,
    list_all_images,
    list_sub_folders,
    load_cache,
    safe_filename,
    save_cache,
)


class DefaultDataset(Dataset):
    def __init__(self):
        pass

    def load_image(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class GeoDataset(Dataset):
    def __init__(self):
        pass

    def load_geodata(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
