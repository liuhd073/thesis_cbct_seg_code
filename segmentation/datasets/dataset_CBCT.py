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


class CBCTDataset(Dataset):
    def __init__(self, files, n_slices=21, transform=None, cachedir="/cache"):
        self.n_slices = n_slices
        self.image_shapes = {tup[0]: (tup[1][0], tup[1][1], tup[1][2]) for tup in files}
        self.data = {tup[0]: (tup[2], tup[3]) for tup in files}
        self.patients = list(self.image_shapes.keys())
        self.transform = transform
        print(self.image_shapes[self.patients[0]], self.image_shapes[self.patients[0]][-3])
        self.cumulative_imshapes = [0] + list(
            np.cumsum([self.image_shapes[patient][-3] for patient in self.patients])
        )
        self.cachedir = Path(cachedir)
        self.current_patient = None

    def __len__(self):
        return sum([self.image_shapes[patient][-3] for patient in self.patients])

    def _get_segmentation(self, segmentations):
        if len(segmentations) == 2:
            seg_bladder = read_image(segmentations[0], no_meta=True)
            seg_cervix_uterus = read_image(segmentations[1], no_meta=True)

        seg_bladder = crop_to_bbox(seg_bladder, (0, 0, 0, seg_bladder.shape[0], 512, 512))
        seg_cervix_uterus = crop_to_bbox(seg_cervix_uterus, (0, 0, 0, seg_cervix_uterus.shape[0], 512, 512))
        all_segs = seg_bladder + seg_cervix_uterus
        other = all_segs < 1
        segs = [seg_bladder, seg_cervix_uterus, other]
        segmentation = np.stack(segs).astype(int)
        return segmentation

    def _load_image(self, patient):
        cache_fn = self.cachedir / f"{patient}_CT1"
        cache_fn_seg = self.cachedir / f"{patient}_CT1_seg"
        if cache_fn.exists() and cache_fn_seg.exists():
            image = read_object(cache_fn)
            segmentation = read_object(cache_fn_seg)
        else:
            image_path, segmentation_paths = self.data[patient]
            image = read_image(image_path, no_meta=True)
            segmentation = self._get_segmentation(segmentation_paths)
            if len(image.shape) == 3:
                # add "channels" dimension if it is not present
                image = np.expand_dims(image, axis=0)
        return image, segmentation

    def _find_image_from_index(self, i):
        """
        Finds the image containing slice "idx", given that
        indices acumulate over images
        """
        # -1 to compensate for the added [0] to cumulative_imshapes in __init__()
        p_idx = bisect_left(self.cumulative_imshapes, i + 1) - 1
        assert p_idx < len(
            self.patients
        ), f"Illegal slice accessed in dataset! Index {i} exceeds size of dataset."
        return self.patients[p_idx], i - self.cumulative_imshapes[p_idx]

    def __getitem__(self, i):
        patient, slice_idx = self._find_image_from_index(i)
        if self.current_patient != patient:
            self.image, self.segmentation = self._load_image(patient)
            self.current_patient = patient
        
        start = 0
        middle_slice = self.n_slices // 2

        im_slice = crop_to_bbox(self.image, (0, slice_idx - middle_slice, start, start, 1, self.n_slices, 512, 512))
        seg_slice = crop_to_bbox(self.segmentation, (0, slice_idx, 0, 0, 3, 1, 512, 512))
        
        sample = {"image": im_slice.squeeze(0), "target": seg_slice.squeeze(1)}
        sample = self.transform(sample)
        im_slice = np.expand_dims(sample["image"],0)
        seg_slice = np.expand_dims(sample["target"],1)
        
        assert (
            0 not in seg_slice.shape
        ), f"Segmentation slice has dimension of size 0: {seg_slice.shape}"

        return im_slice, seg_slice
