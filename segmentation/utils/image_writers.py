# encoding: utf-8
__author__ = 'Jonas Teuwen'
import os
import SimpleITK as sitk
import time
from glob import glob
from .image_readers import resample_sitk_image


def write_image(data, filepath, compression=True, metadata=None, resample=False):
    """
    Parameters
    ----------
    data
    filepath : either a path or folder.
        If this is a folder, the output will be written as IMG-<SeriesNumber>-<SliceNumber>.dcm where series number
        is determined by the other files in the folder.
    compression : bool
        Use compression. When the filepath ends with .nii.gz compression is always enabled
    metadata : dict
        Dictionary containing keys such as spacing, direction and origin. If there is a series_description,
        this will be used when writing dicom output
    resample : str
        Will resample the data prior to writing, resampling data should be available in the metadata dictionary.
    Returns
    -------
    None
    TODO: Better catching of SimpleITK errors
    """

    possible_exts = ['nrrd', 'mhd', 'mha', 'nii', 'nii.gz', 'dcm']
    sitk_image = sitk.GetImageFromArray(data)
    # We need to set spacing, otherwise things go wrong.
    sitk_image.SetSpacing(metadata['spacing'])

    if resample and metadata['orig_spacing'] != metadata['spacing']:
        sitk_image, _ = resample_sitk_image(sitk_image, spacing=metadata['orig_spacing'], interpolator=resample)

    if metadata:
        if 'origin' in metadata:
            sitk_image.SetOrigin(metadata['origin'])
        if 'direction' in metadata:
            sitk_image.SetDirection(metadata['direction'])

    if any([filepath.endswith(_) for _ in possible_exts]):
        try:
            sitk.WriteImage(sitk_image, filepath, True if filepath.endswith('nii.gz') else compression)
        except RuntimeError as e:
            error_str = str(e)
            if error_str.startswith('Exception thrown in SimpleITK WriteImage'):
                if f'Write: Error writing {filepath}' in error_str:
                    raise RuntimeError(f'Cannot write to {filepath}.')
            else:
                raise RuntimeError(e)

    elif os.path.isdir(filepath):
        assert data.ndim == 3, 'For dicom series, only 3D data is supported.'

        series_in_directory = [int(_.split('-')[1]) for _ in glob(os.path.join(filepath, 'IMG-*.dcm'))]
        curr_series = max(series_in_directory)

        modification_time = time.strftime('%H%M%S')
        modification_date = time.strftime('%Y%m%d')
        direction = sitk_image.GetDirection()
        series_tag_values = [
            ('0008|0031', modification_time),  # Series Time
            ('0008|0021', modification_date),  # Series Date
            ('0008|0008', 'DERIVED\\SECONDARY'),  # Image Type
            ('0020|000e', '1.2.826.0.1.3680043.2.1125.' + modification_date + '.1' + modification_time),
            # Series Instance UID
            ('0020|0037', '\\'.join(
                map(str, (
                    direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                    direction[1], direction[4], direction[7])))),
            ('0008|103e', metadata.get('series_description', f'MAPPER OUTPUT {curr_series}'))  # Series Description
        ]
        writer = sitk.ImageFileWriter()
        if compression:
            writer.SetUseCompression()
        # Use the study/series/frame of reference information given in the meta-data dictionary
        writer.KeepOriginalImageUIDOn()
        for idx in range(sitk_image.GetDepth()):
            image_slice = sitk_image[:, :, idx]
            # Set tags specific for series
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            # Set tags specific per slice.
            image_slice.SetMetaData('0008|0012', time.strftime('%Y%m%d'))  # Instance Creation Date
            image_slice.SetMetaData('0008|0013', time.strftime('%H%M%S'))  # Instance Creation Time
            image_slice.SetMetaData('0020|0032', '\\'.join(
                map(str, sitk_image.TransformIndexToPhysicalPoint((0, 0, idx)))))  # Image Position (Patient)
            image_slice.SetMetaData('0020|0013', str(idx))  # Instance Number

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            writer.SetFileName(f'IMG-{curr_series:03d}-{idx:03d}.dcm')
            writer.Execute(image_slice)
    else:
        raise ValueError(f'Filename extension has to be one of {possible_exts}')