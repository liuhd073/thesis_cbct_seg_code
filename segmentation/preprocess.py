'''
Author: Tessa Wagenaar
'''

import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from skimage.transform import rotate, rescale
import random

from utils.bbox import extend_bbox, crop_to_bbox, combine_bbox
MIN_HV = -100
MAX_HV = 300


class ClipAndNormalize(object):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, sample):
        sample["image"] = np.clip(sample["image"], self.minimum, self.maximum)
        sample["image"] = (sample["image"] - self.minimum) / (self.maximum - self.minimum)
        return sample


class NormalizeHV(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample["image"] = (sample["image"] - MIN_HV) / (MAX_HV - MIN_HV)
        return sample


class NormalizeIMG(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample["image"]
        sample["image"] = (image - image.min()) / (image.max() - image.min())
        return sample


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample


class RandomTransform(object):
    """Select a transform randomly from a list"""
    def __init__(self, transforms, choose_weight=None):
        """
        Given a weight, a transform is chosen from a list.
        Parameters
        ----------
        transforms : list
        choose_weight : list or np.ndarray
        """
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        self.choose_weight = choose_weight

        self._num_transforms = len(transforms)

    def __call__(self, sample):
        if self.choose_weight:
            idx = random.choices(range(self._num_transforms), self.choose_weight)[0]
        else:
            idx = np.random.randint(0, self._num_transforms)
        transform = self.transforms[idx]
        sample = transform(sample)  # pylint: disable=not-callable
        return sample


class RandomFlip(object):
    def __init__(self, prob=0.5, axis=0):
        super().__init__()
        self.prob = prob
        self.axis = axis

    def __call__(self, sample):
        flip = np.random.rand() < self.prob
        if flip:
            sample['image'] = np.ascontiguousarray(np.flip(sample['image'], axis=self.axis))
            sample['target'] = np.ascontiguousarray(np.flip(sample['target'], axis=self.axis))
        return sample


class GaussianAdditiveNoise(object):
    """Add Gaussian noise to the input image.
    Examples
    --------
    The following transform could be used to add Gaussian additive noise with 20 HU to the image, and subsequently clip
    to [-300, 100]HU and rescale this to [0, 1].
    >>> transform = Compose([GaussianAdditiveNoise(0, 20), ClipAndScale([-300, 100], [-300, 100], [0, 1])])
    """
    def __init__(self, mean, stddev):
        """
        Adds Gaussian additive noise to the input image.
        Parameters
        ----------
        mean : float
        stddev : float
        """
        self.mean = mean
        self.stddev = stddev

    def apply_transform(self, data):
        return data + np.random.normal(loc=self.mean, scale=self.stddev, size=data.shape)

    def __call__(self, sample):
        sample['image'] = self.apply_transform(sample['image'])
        return sample


class RandomZoom(object):
    def __init__(self, zoom_range, plane='inplane'):
        super().__init__()
        self.zoom_factor = np.random.uniform(1 - zoom_range, 1)
        self.plane = plane

    def __call__(self, sample):
        image = sample['image']
        target = sample['target']

        if self.plane not in ['inplane']:
            raise NotImplementedError()

        image = np.swapaxes(image, 0, 2)
        target = np.swapaxes(target, 0, 2)

        image_resized = np.swapaxes(
            rescale(
                image, self.zoom_factor, order=1, mode='constant',
                clip=False, multichannel=True, anti_aliasing=False), 2, 0)

        target_resized = np.swapaxes(
            rescale(
                target, self.zoom_factor, order=0, mode='constant',
                clip=False, multichannel=True, preserve_range=True, anti_aliasing=False), 2, 0)

        final_image_size = np.asarray(image.shape)
        final_target_size = np.asarray(target.shape)

        image_crop_offset = (np.asarray(image_resized.shape) - final_image_size) // 2
        target_crop_offset = (np.asarray(target_resized.shape) - final_target_size) // 2

        image = crop_to_bbox(image_resized, combine_bbox(image_crop_offset, final_image_size))
        target = crop_to_bbox(target_resized, combine_bbox(target_crop_offset, target_crop_offset))

        sample['image'] = image
        sample['target'] = target
        return sample

