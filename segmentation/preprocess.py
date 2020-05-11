'''
Author: Tessa Wagenaar
'''

import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from skimage.transform import rotate, rescale
import random
import cv2
import time
from utils.bbox import extend_bbox, crop_to_bbox, combine_bbox
cv2.ocl.setUseOpenCL(False)


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample


class ClipAndNormalize(object):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, sample):
        sample["image"] = np.clip(sample["image"], self.minimum, self.maximum)
        sample["image"] = (sample["image"] - self.minimum) / float(self.maximum - self.minimum)
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


class RandomElastic(object):
    def __init__(self, in_shape, std=0.6, sigma=0, scale=0.5):
        """
        Elastic deformation in 2D or 3D per slice. For 3D volumes (z, x, y) keeps z axis unperturbed.
        """
        super().__init__()
        self.in_shape = in_shape
        self.mask = True

        # Make odd
        blur_size = int(std * np.max(in_shape)) | 1
        self.rand_x = scale * cv2.GaussianBlur(
            (np.random.uniform(size=in_shape[-2:]) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * 2 * in_shape[1 if len(in_shape) == 3 else 0]
        self.rand_y = scale * cv2.GaussianBlur(
            (np.random.uniform(size=in_shape[-2:]) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * 2 * in_shape[2 if len(in_shape) == 3 else 1]

    def __call__(self, sample):
        image = sample['image']
        target = sample['target']
        target_shape = target.shape[::-1][:-1]
        time.sleep(1)
        grid_x, grid_y = np.meshgrid(*[np.arange(_) for _ in target_shape])

        grid_x = (grid_x + self.rand_x).astype(np.float32)
        grid_y = (grid_y + self.rand_y).astype(np.float32)

        if len(target.shape) == 2:
            raise NotImplementedError()  # Do not loop over the depth in this case

        image_moved = [
            cv2.remap(image[i, :, :], grid_x, grid_y, borderMode=cv2.BORDER_REFLECT_101,
                      interpolation=cv2.INTER_LINEAR) for i in range(image.shape[0])
        ]

        target_moved = [
            cv2.remap(target[i, :, :].astype(np.float32), grid_x, grid_y, borderMode=cv2.BORDER_REFLECT_101,
                      interpolation=cv2.INTER_LINEAR) for i in range(target.shape[0])]
        sample['image'] = np.stack(np.rint(image_moved), axis=0).astype(np.float32)
        sample['target'] = np.stack(np.rint(target_moved), axis=0).astype(np.uint8)

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


class ToTensor(object):
    def __init__(self, target_is_float=False, add_channel=True):
        """
        Parameters
        ----------
        target_is_float : bool
            Typecast the output as float, e.g., for regression problems
        add_channel : bool
            Add an extra first axis if previously no channels are given.
        """
        self.add_channel = add_channel
        self.target_is_float = target_is_float

    def __call__(self, sample):
        image = sample['image']
        if self.add_channel:
            image = image[np.newaxis, ...]

        target = sample['target']

        image_tensor = torch.from_numpy(image).float()

        if self.target_is_float:
            target_tensor = torch.from_numpy(target).float()
        else:
            target_tensor = torch.from_numpy(target).long()

        sample['image'] = image_tensor
        sample['target'] = target_tensor
        return sample