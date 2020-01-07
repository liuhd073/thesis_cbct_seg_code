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


class CervixDataset(Dataset):
    def __init__(self, patients, root_dir, image_shapes, conebeams=True, shuffle=False):
        super(CervixDataset, self).__init__()
        self.conebeam = conebeams
        self.CBCTs = defaultdict(list)
        self.CTs = defaultdict(list)
        self.root_dir = root_dir
        self.patients = patients
        if self.conebeam:
            self.patients = list(image_shapes.keys())
            self.get_CBCTs()
        else:
            self.get_CTs()

        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.patients)
        self.image_shapes = image_shapes
        self.patient_idx = 0
        self.total = 0 # count for the sum of slices in _previous_ images (so don't count the current)
        if self.conebeam:
            self.n_current_slices = self.image_shapes[self.patients[self.patient_idx]][-2]
        else:    
            self.n_current_slices = self.image_shapes[self.patients[self.patient_idx]][-3]
        self.image, self.segmentation = self._load_image(self.patient_idx)
        self._update_random_list()
        

    def __len__(self):
        if self.conebeam:
            return sum([shape[-2] for shape in self.image_shapes.values()])
        return sum([shape[-3] for shape in self.image_shapes.values()])

    def get_CTs(self):
        for patient in self.patients:
            imgs = glob.glob(os.path.join(self.root_dir, patient, "CT*.nii"))
            for img in imgs:
                m = re.search("CT[0-9]+", img)
                n = m.group(0).lower()
                segmentations = glob.glob(os.path.join(self.root_dir, patient, "*_{}.nii".format(n)))
                if len(segmentations) > 0:
                    self.CTs[patient].append((img, segmentations))

    def get_CBCTs(self):
        for patient in self.patients:
            p = patient.split("\\")
            images = glob.glob(os.path.join(self.root_dir, p[0], p[1] + ".nii"))
            for cbct in images:
                m = re.search("X[0-9]+", cbct)
                n = m.group(0).lower()
                segmentations = glob.glob(os.path.join(self.root_dir, patient[:-4], "*_{}.nii".format(n)))
                if len(segmentations) > 0:
                    self.CBCTs[patient].append((cbct, segmentations))

    def _get_segmentation(self, segmentations):
        segs = []
        all_segs = None
        for seg in segmentations:
            Y = read_image(seg, no_meta=True)
            segs.append(Y)
            if all_segs is None:
                all_segs = Y.copy()
            else: 
                all_segs += Y

        other = all_segs < 1
        segs.append(other)
        segmentation = np.stack(segs).astype(int)
        return segmentation

    def _load_image(self, patient_idx):
        patient = self.patients[patient_idx]
        if self.conebeam:
            images = self.CBCTs[patient]
        else:
            images = self.CTs[patient]

        image_path, segmentations = images[0]
        segmentation = self._get_segmentation(segmentations)
        print('loading', image_path)

        image = read_image(image_path, no_meta=True)
        assert image.shape == segmentation.shape[1:], "image and segmentation should be of same shape in dataset!"
        if len(image.shape) == 3:
            # add "channels" dimension if it is not present
            image = np.expand_dims(image, axis=0)

        image = (image - image.min()) / image.max()
        if self.conebeam:
            image = np.swapaxes(image, 1, 2)
            print(segmentation.shape)
            segmentation = np.swapaxes(segmentation, 1, 2)
            print(segmentation.shape)
        return image, segmentation

    def _update_random_list(self):
        self.random_order = list(range(self.n_current_slices))
        np.random.shuffle(self.random_order)

    def __getitem__(self, i):
        # Assuming i is incremental (not random)
        if i - self.total == self.n_current_slices:
            # update image and segmentation
            self.total += self.n_current_slices
            self.patient_idx += 1
            if self.conebeam:
                self.n_current_slices = self.image_shapes[self.patients[self.patient_idx]][-2]
            else:    
                self.n_current_slices = self.image_shapes[self.patients[self.patient_idx]][-3]
            self.image, self.segmentation = self._load_image(self.patient_idx)
            if self.shuffle:
                self._update_random_list()

        slice_idx = i - self.total
        if self.shuffle:
            slice_idx = self.random_order[slice_idx]
        im_slice = crop_to_bbox(self.image, (0, slice_idx-10, 0, 0, 1, 21, 512, 512))
        seg_slice = self.segmentation[:, slice_idx:slice_idx+1, :, :]
        if self.conebeam:
            seg_slice = crop_to_bbox(seg_slice, (0, slice_idx, 0, 0, 3, 1, 512, 512))

        return im_slice, seg_slice

    
