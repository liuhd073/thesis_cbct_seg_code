"""
Author: Tessa Wagenaar
"""

from bisect import bisect_left
from pathlib import Path
from torch.utils.data import Dataset
from utils.bbox import crop_to_bbox
from utils.image_readers import read_image
from utils.io import read_object, save_object
from preprocess import NormalizeIMG

import numpy as np


class CBCTDataset(Dataset):
    def __init__(self, files, n_slices=21, transform=None, cachedir="/cache"):
        self.n_slices = n_slices
        self.image_shapes = {tup[0]: (tup[1][1], tup[1][0], tup[1][2]) for tup in files}
        print(list(self.image_shapes.values())[0])
        self.data = {tup[0]: (tup[2], tup[3]) for tup in files}
        self.patients = list(self.image_shapes.keys())
        self.transform = transform
        self.cumulative_imshapes = [0] + list(
            np.cumsum([self.image_shapes[patient][-3] for patient in self.patients])
        )
        self.cachedir = Path(cachedir)
        self.current_patient = None

    def __len__(self):
        return sum([self.image_shapes[patient][-3] for patient in self.patients])

    def _get_segmentation(self, segmentations):
        if len(segmentations) == 2:
            seg_bladder = read_image(segmentations[0], no_meta=True) #, spacing=(0.9765625, 0.9765625, 5), interpolator='nearest')
            seg_cervix_uterus = read_image(segmentations[1], no_meta=True) #, spacing=(0.9765625, 0.9765625, 5), interpolator='nearest')
            # all_segs = seg_bladder + seg_cervix_uterus

        print("Segmentation shape:", seg_bladder.shape)
        start = int((seg_bladder.shape[1] - 512) / 2)
        # seg_bladder = crop_to_bbox(seg_bladder, (0, start, start, seg_bladder.shape[0], 512, 512))
        # seg_cervix_uterus = crop_to_bbox(seg_cervix_uterus, (0, start, start, seg_cervix_uterus.shape[0], 512, 512))
        # all_segs = crop_to_bbox(all_segs, (0, start, start, all_segs.shape[0], 512, 512))
        all_segs = seg_bladder + seg_cervix_uterus
        other = all_segs < 1
        segs = [seg_bladder, seg_cervix_uterus, other]
        segmentation = np.stack(segs).astype(int)
        print("Final segmentation shape:", segmentation.shape)
        return segmentation.swapaxes(1,2)

    def _load_image(self, patient):
        cache_fn = self.cachedir / f"{patient}_CT1"
        cache_fn_seg = self.cachedir / f"{patient}_CT1_seg"
        # print("Loading:", cache_fn)
        if cache_fn.exists() and cache_fn_seg.exists():
            image = read_object(cache_fn)
            segmentation = read_object(cache_fn_seg)
        else:
            image_path, segmentation_paths = self.data[patient]
            print(image_path, segmentation_paths)
            image = read_image(image_path, no_meta=True) #, spacing=(0.9765625, 0.9765625, 5))
            print("Image shape:", image.shape)
            image = image.swapaxes(0,1)
            print("Image shape:", image.shape)
            
            # image = crop_to_bbox(image, (0, start, start, image.shape[0], 512, 512))
            # if patient.isdigit():
            image -= 1000
            #     print("SUBSTRACT 1000")
            print(image.min(), image.max())
            segmentation = self._get_segmentation(segmentation_paths)
            # assert (
            #     image.shape == segmentation.shape[1:]
            # ), "image and segmentation should be of same shape in dataset!"
            if len(image.shape) == 3:
                # add "channels" dimension if it is not present
                image = np.expand_dims(image, axis=0)
                # segmentation = np.expand_dims(segmentation, axis=0)
            # if self.preprocess:
            #     image = self.preprocess(image)
            # save_object(image, cache_fn)
            # save_object(segmentation, cache_fn_seg)
        # image = image.swapaxes(0,1)
        # image = segmentation.swapaxes(0,1)
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
        
        print("Image and segmentation shape:", self.image.shape, self.segmentation.shape)

        sample = {"image": self.image, "target": self.segmentation}
        sample = self.transform(sample)
        image = sample["image"]
        segmentation = sample["target"]

        start = int((self.image_shapes[patient][1] - 512) / 2)
        # start = 0
        middle_slice = self.n_slices // 2

        im_slice = crop_to_bbox(image, (0, slice_idx - middle_slice, start, start, 1, self.n_slices, 512, 512))
        seg_slice = crop_to_bbox(segmentation, (0, slice_idx, start, start, 3, 1, 512, 512))

        assert (
            0 not in seg_slice.shape
        ), f"Segmentation slice has dimension of size 0: {seg_slice.shape}"

        return im_slice, seg_slice
