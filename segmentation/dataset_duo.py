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
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class DuoDataset(Dataset):
    def __init__(self, files, n_slices=21, transform=None, cachedir="/cache", return_patient=False):
        self.n_slices = n_slices
        self.return_patient = return_patient
        self.image_shapes = {tup[0]: (tup[1][1], tup[1][0], tup[1][2]) for tup in files}
        self.data_cbct = {tup[0]: (tup[2], tup[3], tup[5]) for tup in files}
        self.data_ct = {tup[0]: (tup[4], tup[6]) for tup in files}
        self.patients = list(self.image_shapes.keys())
        self.transform = transform
        self.cumulative_imshapes = [0] + list(
            np.cumsum([self.image_shapes[patient][-3] for patient in self.patients])
        )
        self.cachedir = Path(cachedir)
        self.current_patient = None

    def __len__(self):
        return sum([self.image_shapes[patient][-3] for patient in self.patients])

    def _get_segmentation_ct(self, segmentations):
        if len(segmentations) == 2:
            seg_bladder = read_image(segmentations[0], no_meta=True) 
            seg_cervix_uterus = read_image(segmentations[1], no_meta=True) 
        start = int((seg_bladder.shape[2] - 512) / 2)
        seg_bladder = crop_to_bbox(seg_bladder, (0, start, start, seg_bladder.shape[0], 512, 512))
        seg_cervix_uterus = crop_to_bbox(seg_cervix_uterus, (0, start, start, seg_cervix_uterus.shape[0], 512, 512))
        all_segs = seg_bladder + seg_cervix_uterus
        other = all_segs < 1
        segs = [seg_bladder, seg_cervix_uterus, other]
        segmentation = np.stack(segs).astype(int)
        return segmentation

    def _get_segmentation_cbct(self, segmentations, ct_seg_fn):
        if len(segmentations) == 4:
            bladder_affine = np.loadtxt(str(segmentations[1]))
            cervix_affine = np.loadtxt(str(segmentations[3]))
            seg_bladder = read_image(segmentations[0], no_meta=True, affine_matrix=bladder_affine, ref_fn=ct_seg_fn[0])
            seg_cervix_uterus = read_image(segmentations[2], no_meta=True, affine_matrix=cervix_affine, ref_fn=ct_seg_fn[1]) 
        
        start = int((seg_bladder.shape[2] - 512) / 2)
        seg_bladder = crop_to_bbox(seg_bladder, (0, start, start, seg_bladder.shape[0], 512, 512))
        seg_cervix_uterus = crop_to_bbox(seg_cervix_uterus, (0, start, start, seg_cervix_uterus.shape[0], 512, 512))
        all_segs = seg_bladder + seg_cervix_uterus
        other = all_segs < 1
        segs = [seg_bladder, seg_cervix_uterus, other]
        segmentation = np.stack(segs).astype(int)
        return segmentation

    def _load_images(self, patient):
        image_path, segmentation_paths = self.data_ct[patient]
        ct, meta_ct = read_image(image_path) 
        segmentation_ct = self._get_segmentation_ct(segmentation_paths)
        logger.debug(f"Image and segmentation shapes:\nCT: {ct.shape}\nCT seg: {segmentation_ct.shape}")

        image_path, affine_fn, segmentation_paths = self.data_cbct[patient]
        ct_fn, ct_seg_fn = self.data_ct[patient]
        affine = np.loadtxt(str(affine_fn))
        cbct = read_image(image_path, no_meta=True, affine_matrix=affine, ref_fn=ct_fn)
        segmentation_cbct = self._get_segmentation_cbct(segmentation_paths, ct_seg_fn)
        logger.debug(f"CBCT Shape: {cbct.shape}")
        logger.debug(f"Image and segmentation shapes:\nCBCT: {cbct.shape}\nCBCT seg: {segmentation_cbct.shape}")

        if len(ct.shape) == 3:
            # add "channels" dimension if it is not present
            ct = np.expand_dims(ct, axis=0)
        if len(cbct.shape) == 3:
            # add "channels" dimension if it is not present
            cbct = np.expand_dims(cbct, axis=0)
        return cbct, ct, segmentation_cbct, segmentation_ct

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
            self.cbct, self.ct, self.segmentation_cbct, self.segmentation_ct = self._load_images(patient)
            self.current_patient = patient
        logger.debug(f"GETITEM Image and segmentation shapes:\nCBCT: {self.cbct.shape}\nCBCT seg: {self.segmentation_cbct.shape}")
        
        start_ct = int((self.ct.shape[2] - 512) / 2)
        start_cbct = int((self.cbct.shape[2] - 512) / 2)
        middle_slice = self.n_slices // 2

        ct_slice = crop_to_bbox(self.ct, (0, slice_idx - middle_slice, start_ct, start_ct, 1, self.n_slices, 512, 512))
        ct_seg_slice = crop_to_bbox(self.segmentation_ct, (0, slice_idx, 0, 0, 3, 1, 512, 512))
        cbct_slice = crop_to_bbox(self.cbct, (0, slice_idx - middle_slice, start_cbct, start_cbct, 1, self.n_slices, 512, 512))
        cbct_seg_slice = crop_to_bbox(self.segmentation_cbct, (0, slice_idx, 0, 0, 3, 1, 512, 512))
        
        ct_sample = {"image": ct_slice.squeeze(0), "target": ct_seg_slice.squeeze(1)}
        ct_sample = self.transform(ct_sample)
        ct_slice = np.expand_dims(ct_sample["image"],0)
        ct_seg_slice = np.expand_dims(ct_sample["target"],1)

        cbct_sample = {"image": cbct_slice.squeeze(0), "target": cbct_seg_slice.squeeze(1)}
        cbct_sample = self.transform(cbct_sample)
        cbct_slice = np.expand_dims(cbct_sample["image"],0)
        cbct_seg_slice = np.expand_dims(cbct_sample["target"],1)
       
        assert (
            0 not in ct_seg_slice.shape and 0 not in cbct_seg_slice.shape
        ), f"Segmentation slice has dimension of size 0: {ct_seg_slice.shape}"

        if self.return_patient:
            return patient, cbct_slice, ct_slice, cbct_seg_slice, ct_seg_slice
        return cbct_slice, ct_slice, cbct_seg_slice, ct_seg_slice
