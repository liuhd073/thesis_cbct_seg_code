"""
Copyright (c) Nikita Moriakov and Jonas Teuwen
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import matplotlib.pyplot as plt
from skimage.feature.peak import peak_local_max
from skimage.measure import find_contours
from skimage.morphology import disk, closing
from matplotlib.ticker import NullLocator
import matplotlib.patches as mpatches
import numpy.ma as ma
import io
import PIL


def plot_2d(image, mask=None, bboxes=None,
            overlay=None, linewidth=2, mask_color='r', mask_threshold=0.5, 
            bbox_color='b', overlay_cmap='jet', overlay_threshold=0.1,
            overlay_alpha=0.1, overlay_local_max_min_distance=75,
            overlay_local_max_color='r', overlay_contour_color='g'):
    """
    Plot image with contours
    Parameters
    ----------
    image
    mask
    bboxes
    overlay
    linewidth
    mask_color
    bbox_color
    overlay_cmap
    overlay_threshold
    overlay_alpha
    overlay_local_max_min_distance
    overlay_local_max_color
    overlay_contour_color
    Returns
    -------
    PIL Image
    """
    dpi = 80
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)

    fig, ax = plt.subplots(1, figsize=figsize)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    cmap = None
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = image[..., 0]
        elif image.shape[0] == 1:
            image = image[0, ...]

    if image.ndim == 2:
        cmap = 'gray'

    ax.imshow(image, cmap=cmap, aspect='equal', extent=(0, width, height, 0))
    ax.set_adjustable('datalim')

    if mask is not None:
        add_2d_contours(mask, ax, linewidth, mask_threshold, mask_color)

    if bboxes is not None:
        for bbox in bboxes:
            add_2d_bbox(bbox, ax, linewidth, bbox_color)

    if overlay is not None:
        add_2d_overlay(overlay, ax, linewidth, threshold=overlay_threshold, cmap=overlay_cmap,
                       alpha=overlay_alpha, contour_color=overlay_contour_color)

    if overlay is not None and overlay_local_max_min_distance:
        coordinates = peak_local_max(overlay, min_distance=overlay_local_max_min_distance,
                                     threshold_abs=overlay_threshold)
        ax.plot(coordinates[:, 1], coordinates[:, 0], overlay_local_max_color + '.', markersize=15, alpha=1)

    fig.gca().set_axis_off()
    fig.gca().xaxis.set_major_locator(NullLocator())
    fig.gca().yaxis.set_major_locator(NullLocator())

    buffer = io.BytesIO()

    fig.savefig(buffer, pad_inches=0, dpi=dpi)
    buffer.seek(0)
    plt.close()

    pil_image = PIL.Image.open(buffer)

    return pil_image


def add_2d_bbox(bbox, ax, linewidth=0.5, color='b'):
    """Add bounding box to the image.
    Parameters
    ----------
    bbox : tuple
        Tuple of the form (row, col, height, width).
    axis : axis object
    linewidth : float
        thickness of the overlay lines
    color : str
        matplotlib supported color string for contour overlay.
    """
    rect = mpatches.Rectangle(bbox[:2][::-1], bbox[3], bbox[2],
                              fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)


def add_2d_contours(mask, axes, linewidth=0.5, threshold=0.5, color='r'):
    """Plot the contours around the `1`'s in the mask
    Parameters
    ----------
    mask : ndarray
        2D binary array.
    axis : axis object
        matplotlib axis object.
    linewidth : float
        thickness of the overlay lines.
    color : str
        matplotlib supported color string for contour overlay.
    TODO: In utils.mask_utils we have function which computes one contour, perhaps these can be merged.
    """
    contours = find_contours(mask, threshold)

    for contour in contours:
        axes.plot(*contour[:, [1, 0]].T, color=color, linewidth=linewidth)


def add_2d_overlay(overlay, ax, linewidth, threshold=0.1, cmap='jet', alpha=0.1, closing_radius=15, contour_color='g'):
    """Adds an overlay of the probability map and predicted regions
    overlay : ndarray
    ax : axis object
       matplotlib axis object
    linewidth : float
       thickness of the overlay lines;
    threshold : float
    cmap : str
       matplotlib supported cmap.
    alpha : float
       alpha values for the overlay.
    closing_radius : int
       radius for the postprocessing closing
    contour_color : str
       matplotlib supported color for the contour.
    """
    if threshold:
        overlay = ma.masked_where(overlay < threshold, overlay)

    ax.imshow(overlay, cmap=cmap, alpha=alpha)

    if contour_color:
        mask = closing(overlay.copy(), disk(closing_radius))
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        add_2d_contours(mask, ax, linewidth=linewidth, color=contour_color)