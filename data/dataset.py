import os

import numpy as np
from PIL import Image
from torch.utils import data

from utils.file import list_all_images, list_sub_folders, exist_cache, load_cache, save_cache, safe_filename


class DefaultDataset(data.Dataset):
    """ No label. """

    def __init__(self, root, transform=None):
        self.samples = list_all_images(root)
        self.samples.sort()
        self.transform = transform

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        return self.load_image(self.samples[index])

    def __len__(self):
        return len(self.samples)
