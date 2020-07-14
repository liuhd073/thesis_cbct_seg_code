# encoding: utf-8
import os

import SimpleITK as sitk
import numpy as np
import tempfile

_DICOM_MODALITY_TAG = '0008|0060'
_DICOM_VOI_LUT_FUNCTION = '0028|1056'
_DICOM_WINDOW_CENTER_TAG = '0028|1050'
_DICOM_WINDOW_WIDTH_TAG = '0028|1051'
_DICOM_WINDOW_CENTER_WIDTH_EXPLANATION_TAG = '0028|1055'


# https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        ref_image=None, affine_matrix=None, fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()
    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: #  8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not ref_image is None: 
        new_origin = ref_image.GetOrigin()
        new_direction = ref_image.GetDirection()
    else: 
        new_origin = orig_origin
        new_direction = orig_direction

    if not ref_image is None:
        new_spacing = ref_image.GetSpacing()
    elif not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing]*num_dim
    else:
        new_spacing = [float(s) if s else orig_spacing[idx] for idx, s in enumerate(spacing)]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    if ref_image is None:
        new_size = orig_size*(orig_spacing/new_spacing)
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        # SimpleITK expects lists, not ndarrays
        new_size = [int(s) if spacing[idx] else int(orig_size[idx]) for idx, s in enumerate(new_size)]
    else: 
        new_size = ref_image.GetSize()

    if not affine_matrix is None and False:
        translation = sitk.TranslationTransform(3)
        translation.SetOffset(affine_matrix[:3,3].ravel())

        affine = sitk.AffineTransform(3)
        affine.SetMatrix(affine_matrix[:3,:3].ravel())

        transform = sitk.Transform(3, sitk.sitkComposite)
        transform.AddTransform(affine)
        transform.AddTransform(translation)
    else:
        transform = sitk.Transform()
    
    resample_filter = sitk.ResampleImageFilter()
    resampled_sitk_image = resample_filter.Execute(
        sitk_image,
        new_size,
        transform,
        sitk_interpolator,
        new_origin,
        new_spacing,
        new_direction,
        fill_value,
        orig_pixelid
    )

    meta = {"orig_spacing": orig_spacing, "orig_origin": orig_origin, "orig_direction": orig_direction, "orig_shape": orig_size}
    return resampled_sitk_image, meta

def read_image(filename, force_2d=False, dtype=None, no_meta=False, affine_matrix=None, ref_fn=None, **kwargs):
    """Read medical image

    Parameters
    ----------
    filename : str
        Path to image, can be any SimpleITK supported filename
    force_2d : bool
        If the image is 2D it can happen the image is presented as 3D but with (height, width, 1),
        this option reduces the image to 2D.
    dtype : dtype
        The requested dtype the output should be cast.
    no_meta : bool
        Do not output metadata

    Returns
    -------
    Image as ndarray and dictionary with metadata.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'{filename} does not exist.')

    new_spacing = kwargs.get('spacing', False)
    if new_spacing and np.all(np.asarray(new_spacing) <= 0):
        new_spacing = False

    if os.path.splitext(filename)[-1].lower() == '.dcm':
        # TODO merge read_dcm in this function
        if new_spacing:
            raise NotImplementedError()
        image, metadata = read_dcm(filename, **kwargs)
    else:
        metadata = {}
        if not ref_fn is None:
            sitk_ref_image = sitk.ReadImage(str(ref_fn))
        else: 
            sitk_ref_image = None
        sitk_image = sitk.ReadImage(str(filename))
        orig_shape = sitk.GetArrayFromImage(sitk_image).shape
        if new_spacing or not (affine_matrix is None):
            sitk_image, meta_data = resample_sitk_image(
                sitk_image,
                spacing=new_spacing,
                interpolator=kwargs.get('interpolator', None),
                ref_image=sitk_ref_image,
                affine_matrix=affine_matrix,
                fill_value=0
            )
            metadata.update(meta_data)
        image = sitk.GetArrayFromImage(sitk_image)
        if force_2d:
            if image.ndim == 3 and image.shape[0] == 1:
                image = image[0]
            else:
                raise ValueError('Can only force image to be 2D when the depth is 1.')

        metadata.update({
            'filename': os.path.abspath(filename),
            'depth': sitk_image.GetDepth(),
            'spacing': sitk_image.GetSpacing(),
            'shape': image.shape,
            'origin': sitk_image.GetOrigin(),
            'direction': sitk_image.GetDirection()
        })

    if dtype:
        image = image.astype(dtype)

    if no_meta:
        return image

    return image, metadata


def _apply_window_level(sitk_image, voi_lut_fn='LINEAR', out_range=[0, 255], which_explanation=0):
    """Apply window and level to a SimpleITK image.

    Parameters
    ----------
    sitk_image : SimpleITK image instance
    out_range : tuple or list of new range

    Returns
    -------
    SimpleITK image
    """

    center = sitk_image.GetMetaData(
        _DICOM_WINDOW_CENTER_TAG).strip()
    width = sitk_image.GetMetaData(
        _DICOM_WINDOW_WIDTH_TAG).strip()

    try:
        explanation = sitk_image.GetMetaData(_DICOM_WINDOW_CENTER_WIDTH_EXPLANATION_TAG).strip()
    except RuntimeError:
        explanation = '\\'*len(center.split('\\'))

    exp_split = explanation.split('\\')
    if len(exp_split) > 1:
        c_split = center.split('\\')
        w_split = width.split('\\')
        if isinstance(which_explanation, int):
            idx = which_explanation
        else:
            idx = exp_split.index(which_explanation)
        center = float(c_split[idx])
        width = float(w_split[idx])
    else:
        center = float(center)
        width = float(width)

    if voi_lut_fn == 'LINEAR':
        lower_bound = center - (width - 1)/2
        upper_bound = center + (width - 1)/2
    elif voi_lut_fn == 'SIGMOID':
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        spacing = sitk_image.GetSpacing()
        arr = sitk.GetArrayFromImage(sitk_image)
        arr = 1.0 / (1 + np.exp(-4 * (arr - center)/width))
        sitk_image = sitk.GetImageFromArray(arr)
        sitk_image.SetOrigin(origin)
        sitk_image.SetDirection(direction)
        sitk_image.SetSpacing(spacing)
        # TODO: Check if this makes sense.
        lower_bound = arr.min()
        upper_bound = arr.max()
    else:
        raise ValueError(f'{voi_lut_fn} is not implemented.')

    sitk_image = sitk.IntensityWindowing(
        sitk_image, lower_bound, upper_bound,
        out_range[0], out_range[1])
    # Recast after intensity windowing.
    if (out_range[0] >= 0) and (out_range[1] <= 255):
        pass
    else:
        raise NotImplementedError('Only uint8 supported.')

    sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
    return sitk_image


def read_dcm(filename, window_leveling=True, dtype=None, **kwargs):
    """Read single dicom files. Tries to apply VOILutFunction if available.
    Check if the file is a mammogram or not.

    Parameters
    ----------
    filename : str
        Path to dicom file
    window_leveling : bool
        Whether to apply the window level settings.
    dtype : dtype
        The type the output should be cast.

    Returns
    -------
    Image as ndarray and dictionary with metadata

    TODO: Rename to read_mammo and rebuild the read_dcm function.
    TODO: Seperate function to only read the dicom header.
    """
    if not os.path.exists(filename):
        raise IOError(f'{filename} does not exist.')

    if not os.path.splitext(str(filename))[1] == '.dcm':
        raise ValueError(f'{filename} should have .dcm as an extension')

    # SimpleITK has issues with unicode string names.
    sitk_image = sitk.ReadImage(str(filename))
    try:
        modality = sitk_image.GetMetaData(_DICOM_MODALITY_TAG).strip()
    except RuntimeError as e:  # The key probably does not exist
        modality = None
        raise ValueError(f'Modality tag {_DICOM_MODALITY_TAG} does not exist: {f}')

    try:
        voi_lut_func = sitk_image.GetMetaData(
            _DICOM_VOI_LUT_FUNCTION).strip()
        if voi_lut_func == '':
            voi_lut_func = 'LINEAR'
    except RuntimeError:
        voi_lut_func = 'LINEAR'

    # Check if kwargs contains extra dicom tags
    dicom_keys = kwargs.get('dicom_keys', None)
    extra_metadata = {}
    if dicom_keys:
        metadata_keys = sitk_image.GetMetaDataKeys()
        for k, v in dicom_keys:
            # VOILUTFunction if missing should be interpreted as 'IDENTITY'
            # per http://dicom.nema.org/medical/dicom/2017a/output/chtml/part03/sect_C.11.2.html
            if v == '0028|1056' and v not in metadata_keys:  # VOILUTFunction:
                extra_metadata[k] = 'IDENTITY'
            else:
                extra_metadata[k] = None if v not in metadata_keys else sitk_image.GetMetaData(v).strip()

    # This needs to be done after reading all tags.
    # The DICOM tags are lost after this operation.
    if window_leveling:
        try:
            sitk_image = _apply_window_level(sitk_image, voi_lut_func)
        except NotImplementedError as e:
            raise NotImplementedError(f'{filename}: {e}')

    data = sitk.GetArrayFromImage(sitk_image)
    if dtype:
        data = data.astype(dtype)

    metadata = {}
    metadata.update(extra_metadata)
    metadata['filename'] = os.path.abspath(filename)
    metadata['depth'] = sitk_image.GetDepth()
    metadata['modality'] = 'n/a' if not modality else modality
    metadata['spacing'] = sitk_image.GetSpacing()
    metadata['origin'] = sitk_image.GetOrigin()
    metadata['direction'] = sitk_image.GetDirection()
    metadata['shape'] = data.shape

    if modality == 'MG':
        # If modality is MG the image can be a DBT image.
        # If the image is true mammogram, we reshape.
        if metadata['depth'] == 1:
            del metadata['depth']
            data = data[0]
            metadata['spacing'] = metadata['spacing'][:-1]
            metadata['shape'] = metadata['shape'][1:]
    else:
        raise NotImplementedError(f'{filename}: Modality {modality} not implemented.')

    return data, metadata


def read_dcm_series(path, series_id=None, filenames=False, return_sitk=False):
    """Read dicom series from a folder. If multiple dicom series are availabe in the folder,
    no image is returned. The metadata dictionary then contains the SeriesIDs which can be selected.

    Parameters
    ----------
    path : str
        path to folder containing the series
    series_id : str
        SeriesID to load
    filenames : str
        If filenames is given then series_id is ignored, and assume that there is one series and these files are loaded.
    return_sitk : bool
        If true, the original SimpleITK image will also be returned

    Returns
    -------
    metadata dictionary and image as ndarray.

    TODO
    ----
    Catch errors such as
    WARNING: In /tmp/SimpleITK-build/ITK/Modules/IO/GDCM/src/itkGDCMSeriesFileNames.cxx, line 109
    GDCMSeriesFileNames (0x4a6e830): No Series were found
    """

    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a directory')

    metadata = {'filenames': []}

    if filenames:
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(filenames[0])
        file_reader.ReadImageInformation()
        series_id = file_reader.GetMetaData('0020|000e')
        with tempfile.TemporaryDirectory() as tmpdir_name:
            for f in filenames:
                os.symlink(os.path.abspath(f), os.path.join(tmpdir_name, os.path.basename(f)))
            sorted_filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(tmpdir_name, series_id)
            metadata['filenames'].append(sorted_filenames)
    else:
        reader = sitk.ImageSeriesReader()
        series_ids = list(reader.GetGDCMSeriesIDs(str(path)))

        for series_id in series_ids:
            sorted_filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_id)
            metadata['filenames'].append(sorted_filenames)
            # TODO: Get series description.

        if len(series_ids) > 1 and not series_id:
            image = None
            metadata['series_ids'] = series_ids

            return image, metadata

    metadata['series_ids'] = series_ids
    sitk_image = sitk.ReadImage(sorted_filenames)

    metadata['filenames'] = [sorted_filenames]
    metadata['depth'] = sitk_image.GetDepth()
    metadata['spacing'] = tuple(sitk_image.GetSpacing())
    metadata['origin'] = tuple(sitk_image.GetOrigin())
    metadata['direction'] = tuple(sitk_image.GetDirection())
    data = sitk.GetArrayFromImage(sitk_image)

    if return_sitk:
        return data, sitk_image, metadata

    return data, metadata