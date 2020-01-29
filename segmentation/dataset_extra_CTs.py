import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from utils.image_readers import read_image
from collections import defaultdict
from utils.bbox import crop_to_bbox
import os
import glob
import re


class ExtraCervixDataset(Dataset):
    def __init__(self, root_dir, image_shapes, transform=None, shuffle=False):
        super(ExtraCervixDataset, self).__init__()
        self.CTs = defaultdict(list)
        self.root_dir = root_dir
        self.patients = list(image_shapes.keys())
        self.get_CTs()

        self.transform = transform

        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.patients)
        self.image_shapes = image_shapes
        self.patient_idx = 0
        # count for the sum of slices in _previous_ images (so don't count the current)
        self.total = 0
        self.n_current_slices = self.image_shapes[self.patients[self.patient_idx]][-3]
        # self.image, self.segmentation = self._load_image(self.patient_idx)
        self._update_random_list()

    def __len__(self):
        return sum([shape[-3] for shape in self.image_shapes.values()])

    def get_CTs(self):
        for patient in self.patients:
            path = os.path.join(self.root_dir, patient)
            img = os.path.join(path, "CT.nrrd")
            if "/full/" in img:
                seg_bladder = os.path.join(path, "CT-Bladder_full.nrrd")
                seg_cervix = os.path.join(path, "CT-Cervix_full.nrrd")
                seg_uterus = os.path.join(path, "CT-Uterus_full.nrrd")
            else:
                seg_bladder = os.path.join(path, "CT-Bladder_empty.nrrd")
                seg_cervix = os.path.join(path, "CT-Cervix_empty.nrrd")
                seg_uterus = os.path.join(path, "CT-Uterus_empty.nrrd")

            segmentations = [seg_bladder, seg_cervix, seg_uterus]
            self.CTs[patient].append((img, segmentations))
        print("CTs loaded")


    def _get_segmentation(self, segmentations):
        seg_bladder = read_image(segmentations[0], no_meta=True).copy()
        seg_cervix = read_image(segmentations[1], no_meta=True).copy()
        seg_uterus = read_image(segmentations[2], no_meta=True).copy()
        seg_cervix_uterus = seg_cervix + seg_uterus > 0
        all_segs = seg_bladder + seg_cervix_uterus
        seg_other = all_segs < 1
        segs = [seg_bladder, seg_cervix_uterus, seg_other]
        segmentation = np.stack(segs).astype(int)
        return segmentation

    def _load_image(self, patient_idx):
        patient = self.patients[patient_idx]
        images = self.CTs[patient]
        image_path, segmentations = images[0]
        segmentation = self._get_segmentation(segmentations)
        print('loading', image_path)

        image = read_image(image_path, no_meta=True)
        assert image.shape == segmentation.shape[1:], "image and segmentation should be of same shape in dataset!"
        if len(image.shape) == 3:
            # add "channels" dimension if it is not present
            image = np.expand_dims(image, axis=0)

        return image, segmentation

    def _update_random_list(self):
        self.random_order = list(range(self.n_current_slices))
        np.random.shuffle(self.random_order)

    def __getitem__(self, i):
        if i == 0:
            self.total = 0
            self.patient_idx = 0
            self.n_current_slices = self.image_shapes[self.patients[self.patient_idx]][-3]
            self.image, self.segmentation = self._load_image(self.patient_idx)
            if self.shuffle:
                self._update_random_list()
        # Assuming i is incremental (not random)
        if i - self.total == self.n_current_slices:
            # update image and segmentation
            self.total += self.n_current_slices

            self.patient_idx += 1
            self.n_current_slices = self.image_shapes[self.patients[self.patient_idx]][-3]
            self.image, self.segmentation = self._load_image(self.patient_idx)
            if self.shuffle:
                self._update_random_list()

        slice_idx = i - self.total
        if self.shuffle:
            slice_idx = self.random_order[slice_idx]
        im_slice = crop_to_bbox(
            self.image, (0, slice_idx-10, 0, 0, 1, 21, 512, 512))
        seg_slice = self.segmentation[:, slice_idx:slice_idx+1, :, :]

        if self.transform:
            im_slice = self.transform(im_slice)

        return im_slice, seg_slice
