# encoding: utf-8
import numpy as np


def split_bbox(bbox):
    """Split bbox into coordinates and size

    Parameters
    ----------
    bbox : tuple or ndarray. Given dimension n, first n coordinates are the starting point, the other n the size.

    Returns
    -------
    coordinates and size, both ndarrays.
    """
    bbox = np.asarray(bbox)

    ndim = int(len(bbox) / 2)
    bbox_coords = bbox[:ndim]
    bbox_size = bbox[ndim:]
    return bbox_coords, bbox_size


def combine_bbox(bbox_coords, bbox_size):
    """Combine coordinates and size into a bounding box.

    Parameters
    ----------
    bbox_coords : tuple or ndarray
    bbox_size : tuple or ndarray

    Returns
    -------
    bounding box

    """
    bbox_coords = np.asarray(bbox_coords).astype(int)
    bbox_size = np.asarray(bbox_size).astype(int)
    bbox = tuple(bbox_coords.tolist() + bbox_size.tolist())
    return bbox


def enclosing_bbox(bboxes):
    """
    Compute the smallest enclosing bounding box for the list of boxes given
    Parameters
    ----------
    bboxes : list of bounding boxes

    Returns
    -------
    bounding box
    """

    splitted_boxes = [split_bbox(convert_bbox(bbox)) for bbox in bboxes]
    starting_points = []
    ending_points = []
    for starting_point, ending_point in splitted_boxes:
        starting_points.append(starting_point)
        ending_points.append(ending_point)

    new_starting_point = np.array(starting_points).min(axis=0)
    new_ending_point = np.array(ending_points).max(axis=0)

    return combine_bbox(new_starting_point, new_ending_point - new_starting_point)


def extend_bbox(bbox, extend, retrieve_original=False):
    """Extend bounding box by `extend`. Will enlarge the bounding box by extend and shift by extend // 2.
    If retrieve_original is True will returns the pair (newbbox, oldbbox) of the new and the original bbox in the
    relative coordinates of the new one.

    Parameters
    ----------
    bbox : tuple or ndarray
    extend : tuple or ndarray
    retrieve_original: boolean
    Returns
    -------
    bounding box

    """
    if not np.any(extend):
        return bbox

    bbox_coords, bbox_size = split_bbox(bbox)
    extend = np.asarray(extend)
    newbbox = combine_bbox(bbox_coords - extend // 2, bbox_size + extend)
    if retrieve_original:
        return newbbox, combine_bbox(extend // 2, bbox_size)
    else:
        return newbbox


def convert_bbox(bbox, to='xyXY'):
    """

    Parameters
    ----------
    bbox
    to : str

    Returns
    -------

    """
    if not to in ['xyXY']:
        raise NotImplementedError()

    if to == 'xyXY':
        bbox_coords, bbox_size = split_bbox(bbox)
        bbox_coords2 = np.asarray(bbox_coords) + np.asarray(bbox_size)

        bbox = np.concatenate([bbox_coords, bbox_coords2]).tolist()
    return bbox


def crop_bbox_to_shape(bbox_orig, shape):
    """

    Parameters
    ----------
    bbox_orig : tuple or ndarray
    shape : tuple

    Returns
    -------
    bounding box and offsets needed to add to original bbox to get the cropped one.

    TODO: Efficiency...

    """
    ndim = int(len(bbox_orig) / 2)
    bbox = np.array(bbox_orig)

    bbox[bbox < 0] = 0
    for idx, _ in enumerate(bbox[ndim:]):
        if _ + bbox[idx] > shape[idx]:
            bbox[idx + ndim] = shape[idx] - bbox[idx]

    return bbox


def get_random_shift_bbox(bbox, minoverlap=0.3, exclude=[]):
    """Shift bbox randomly so that all its sides have at least minoverlap fraction of intersection with originial.
    Dimension in exclude are fixed.
    Parameters
    ----------
    bbox: tuple
    minoverlap: number in (0, 1)
    exclude: tuple
    Returns
    -------
    bbox: tuple
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    deltas = [np.floor(val * minoverlap).astype(int) for val in bbox_size]
    out_coords = []
    for i, coord, delta, sz in zip(range(len(bbox_coords)), bbox_coords, deltas, bbox_size):
        if i in exclude:
            out_coords.append(coord)
        else:
            x = np.random.randint(coord - sz + delta + 1, high=(coord + sz - delta - 1), size=1)[0]
            out_coords.append(x)

    return list(out_coords) + list(bbox_size)


def add_dim(bbox, dim_sz, pre=True, coord=0):
    """Add extra dimension to bbox of size dim_sz
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    if pre:
        bbox_coords = [coord] + bbox_coords.tolist()
        bbox_size = [dim_sz] + bbox_size.tolist()
    else:
        bbox_coords = bbox_coords.tolist() + [coord]
        bbox_size = bbox_size.tolist() + [dim_sz]
    return combine_bbox(bbox_coords, bbox_size)


def project_bbox(bbox, exclude=[]):
    """Project bbox by excluding dimensions in exclude
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    out_coords = []
    out_size = []
    for x, d, i in zip(bbox_coords, bbox_size, range(len(bbox_coords))):
        if i not in exclude:
            out_coords.append(x)
            out_size.append(d)
    return out_coords + out_size


def expand_to_multiple(bbox, div=16, exclude=[]):
    """Extend bounding box so that its sides are multiples of given number, unless axis is in exclude.
    Parameters
    ----------
    bbox: list or tuple
        bbox of the form (coordinates, size)
    div: integer
        value which we want the bounding box sides to be multiples of
    exclude: list or tuple
        list of axis which are left unchanged
    Returns
    -------
    bounding box
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    extend = [int(idx not in exclude) * (div - (val % div)) for idx, val in enumerate(bbox_size)]
    return extend_bbox(bbox, extend)


def bounding_box(mask, ignore_axes=[]):
    """
    Computes the bounding box of a mask
    Parameters
    ----------
    mask : array-like
        Input mask
    ignore_axes : list
        Axes to ignore when computing bounding box

    Returns
    -------
    Bounding box
    """
    bbox_coords = []
    bbox_sizes = []
    for idx in range(mask.ndim):
        axis = tuple([i for i in range(mask.ndim) if i != idx])
        if idx in ignore_axes:
            min_val = 0
            max_val = mask.shape[idx]
        else:
            nonzeros = np.any(mask, axis=axis)
            min_val, max_val = np.where(nonzeros)[0][[0, -1]]
        bbox_coords.append(min_val)
        bbox_sizes.append(max_val - min_val)

    return combine_bbox(bbox_coords, bbox_sizes)


def crop_to_bbox(image, bbox, pad_value=0, ignore_first_axis=False):
    """Extract bbox from images, coordinates can be negative.

    Parameters
    ----------
    image : ndarray
       nD array
    bbox : list or tuple
       bbox of the form (coordinates, size),
       for instance (4, 4, 2, 1) is a patch starting at row 4, col 4 with height 2 and width 1.
    pad_value : number
       if bounding box would be out of the image, this is value the patch will be padded with.
    ignore_first_axis : bool
        do not crop along first axis

    Returns
    -------
    ndarray
    """
    # Coordinates, size
    bbox_coords, bbox_size = split_bbox(bbox)

    if ignore_first_axis:
        # TODO: This should be an expand along array function.
        bbox_coords = np.asarray([0] + list(bbox_coords))
        bbox_size = np.asarray([image.shape[0]] + list(bbox_size))

    # Offsets
    l_offset = -bbox_coords.copy()
    l_offset[l_offset < 0] = 0
    r_offset = (bbox_coords + bbox_size) - np.array(image.shape)
    r_offset[r_offset < 0] = 0

    region_idx = [slice(i, j) for i, j
                  in zip(bbox_coords + l_offset,
                         bbox_coords + bbox_size - r_offset)]
    out = image[tuple(region_idx)]

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    patch = pad_value * np.ones(bbox_size, dtype=image.dtype)
    patch_idx = [slice(i, j) for i, j
                 in zip(l_offset, bbox_size - r_offset)]
    patch[tuple(patch_idx)] = out
    return patch
