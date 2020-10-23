"""
Author: Tessa Wagenaar
"""

from bisect import bisect_left
from pathlib import Path
from torch.utils.data import Dataset
from utils.bbox import crop_to_bbox
from utils.image_readers import read_image
from utils.io import read_object, save_object

import numpy as np


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.split = len(dataset1)

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, i):
        if i >= self.split:
            return self.dataset2.__getitem__(i-self.split)
        return self.dataset1.__getitem__(i)