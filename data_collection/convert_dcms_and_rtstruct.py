"""This file converts dcm (for CT) to nrrd"""
import sys
import pydicom as dicom
import argparse
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from skimage.draw import polygon
from itertools import groupby, zip_longest
from collections import defaultdict
from pathlib import Path


_DICOM_MODALITY_TAG = (0x8, 0x60)
_DICOM_REFERENCED_FRAME_OF_REFERENCE_SEQUENCE_TAG = (0x3006, 0x10)
_DICOM_FRAME_OF_REFERENCE_UID_TAG = (0x20, 0x52)
_DICOM_STRUCTURE_SET_RIO_SEQUENCE_TAG = (0x3006, 0x20)
_DICOM_ROI_CONTOUR_SEQUENCE_TAG = (0x3006, 0x39)
_DICOM_ROI_DISPLAY_COLOR_TAG = (0x3006, 0x2a)
_DICOM_REFERENCED_ROI_NUMBER_TAG = (0x3006, 0x84)
_DICOM_ROI_NUMBER_TAG = (0x3006, 0x22)
_DICOM_CONTOUR_SEQUENCE_TAG = (0x3006, 0x40)
_DICOM_CONTOUR_DATA_TAG = (0x3006, 0x50)
_DICOM_ROI_NAME_TAG = (0x3006, 0x26)
_DICOM_STRUCTURE_SET_DATE_TAG = (0x3006, 0x8)
_DICOM_STRUCTURE_SET_TIME_TAG = (0x3006, 0x9)


def grouper(iterable, n):
    """Given a long string it groups in pairs of `n`"""
    args = [iter(iterable)] * n
    return zip_longest(*args)


def has_frame_of_reference_uid(path, frame_of_reference_uid):
    data = dicom.read_file(path)
    return _DICOM_FRAME_OF_REFERENCE_UID_TAG in data and \
        data[_DICOM_FRAME_OF_REFERENCE_UID_TAG].value == frame_of_reference_uid


def contour_world_to_index(contour_data, image):
    return [
        image.TransformPhysicalPointToIndex(point) for point in
        [
            [float(_) for _ in group] for group in grouper(contour_data, 3)
        ]
    ]


def read_rtstruct(image_directory, rtstruct_filename):
    """Reads directory of DICOM files and rstruct file"""
    output_dict = dict()
    output_dict['image_directory'] = image_directory

    # First we read the rstruct file
    if rtstruct_filename:
        rtstruct = dicom.read_file(rtstruct_filename)
        if not rtstruct[_DICOM_MODALITY_TAG].value == 'RTSTRUCT':
            raise ValueError('{} is not an RTSTRUCT.'.format(rtstruct_filename))

        # We get the frame of reference UID as it is possible that there are
        # multiple images in the image directory, for instance CT, MR, etc
        frame_of_reference_uid = rtstruct[
            _DICOM_REFERENCED_FRAME_OF_REFERENCE_SEQUENCE_TAG
        ][0][_DICOM_FRAME_OF_REFERENCE_UID_TAG].value
        output_dict['frame_of_reference_uid'] = frame_of_reference_uid
    else:
        output_dict['frame_of_reference_uid'] = None

    output_dict['rtstruct_filename'] = rtstruct_filename
    # We are interested in the series which corresponds to the frame of
    # reference of the RTSTRUCT. We look for all DICOM series IDs in the
    # directory, and select the one which corresponds to the RTSTRUCT.
    reader = sitk.ImageSeriesReader()
    series = []
    series_ids = reader.GetGDCMSeriesIDs(image_directory)
    for series_id in series_ids:
        filenames = reader.GetGDCMSeriesFileNames(image_directory, series_id)
        # If there is no rtstruct, then there can be at most one series
        if not rtstruct_filename:
            assert len(series_ids) == 1, 'There can be at most one series when no rtstruct is given'
            series = filenames
        else:
            if has_frame_of_reference_uid(filenames[0], frame_of_reference_uid):
                series = filenames
                break

    # We first need to check whether we found a corresponding series.
    if not series:
        raise ValueError(
            '{} does not contain a series with frame of reference'
            'UID {} corresponding to RTSTRUCT {}'.format(image_directory, frame_of_reference_uid, rtstruct_filename))
    output_dict['num_slices'] = len(series)

    # We proceed by reading the actually series.
    reader.SetFileNames(series)
    image = reader.Execute()
    output_dict['sitk_image'] = image

    if rtstruct_filename:
        # Find all ROI names
        roi_names = [structure[_DICOM_ROI_NAME_TAG].value for structure\
                     in rtstruct[_DICOM_STRUCTURE_SET_RIO_SEQUENCE_TAG].value]
        output_dict['roi_names'] = roi_names

        # We start by constructing an empty dictionary with all available ROIs.
        structures = {
            structure[_DICOM_ROI_NUMBER_TAG].value: {
                'roi_name': structure[_DICOM_ROI_NAME_TAG].value,
                'points': []
            } for structure in
            rtstruct[_DICOM_STRUCTURE_SET_RIO_SEQUENCE_TAG].value
        }

        # A check if we actually have a structure set
        if not structures:
            raise ValueError(
                f'{rtstruct_filename} does not contain any ROIs')

        # Next, we fill the points
        for contour in rtstruct[_DICOM_ROI_CONTOUR_SEQUENCE_TAG].value:
            # Each ROI has a number.
            roi_no = contour[_DICOM_REFERENCED_ROI_NUMBER_TAG].value
            try:
                for contour_string in contour[_DICOM_CONTOUR_SEQUENCE_TAG].value:
                    # We can extract the string containing the contour information
                    contour_data = contour_string[_DICOM_CONTOUR_DATA_TAG].value
                    # Convert the contour data to points and store in the structure.
                    structures[roi_no]['points'] += contour_world_to_index(
                        contour_data, image)
            except KeyError:
                # Ignore missing contours
                pass

        # The structures dictionary is slightly inconvenient, but this is
        # unfortunately how it is stored in the RTSTRUCT. We rewrite it here
        new_structures = {}
        for roi_no, roi in structures.items():
            roi_name = roi['roi_name']
            new_structures[roi_name] = {
                'roi_number': roi_no,
                'points': roi['points']
            }
        output_dict['structures'] = new_structures

    return output_dict


class DicomRtstructReader(object):
    def __init__(self, image_folder, rtstruct_filename):
        raw_dict = read_rtstruct(image_folder, rtstruct_filename)
        self.roi_names = raw_dict.get('roi_names', None)
        self.num_slices = raw_dict['num_slices']

        self.image = raw_dict['sitk_image']
        self.structures = raw_dict.get('structures', [])

    def get_roi(self, roi_name):
        assert roi_name in self.roi_names, f'ROI {roi_name} does not exist'
        roi_dict = self.structures[roi_name]

        # Placeholder for ROI
        # TODO: Improve
        roi = np.zeros(
            (self.image.GetDepth(), self.image.GetHeight(), self.image.GetWidth()), dtype=np.uint8)

        for z, points in groupby(roi_dict['points'], key=lambda point: point[2]):
            points_list = list(points)
            y = [point[1] for point in points_list]
            x = [point[0] for point in points_list]
            rr, cc = polygon(y, x)
            roi[z, rr, cc] = 1

        sitk_roi = sitk.GetImageFromArray(roi)
        sitk_roi.SetSpacing(list(self.image.GetSpacing()))
        sitk_roi.SetDirection(self.image.GetDirection())
        sitk_roi.SetOrigin(self.image.GetOrigin())

        return sitk_roi


# def get_scans_and_rtstructs(path):
#     patients = glob(os.path.join(path, '*'))
#     scans = []
#     for patient in patients:
#         if 'nrrds' in os.path.basename(patient):
#             continue
#         scans.append(os.path.basename(patient))

#     return scans


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Parse MRI dataset')

    parser.add_argument(
        'root_dir',
        type=Path,
        help='root to data',)
    parser.add_argument(
        'write_to',
        type=Path,
        help='folder to write output to', )
    parser.add_argument(
        'csv_dir',
        type=Path,
        help='folder with dicom data in csv format', )

    return parser.parse_args()

def read_dataframe(csvs):
    df = pd.concat([pd.read_csv(csv,  dtype={'PatientID':str, 'Rows': pd.Int64Dtype(), 'InstanceNumber': pd.Int64Dtype(), 'Columns': pd.Int64Dtype(), 'SeriesNumber': pd.Int64Dtype()}) for csv in csvs])
    df = df.reset_index()
    df.insert(loc=16, column='ReferencedSeriesInstanceUID', value='')
    df.insert(loc=17, column='CorrespondingRTSTRUCTReferencedFileID', value='')
    rtstructs = df.index[df['Modality'] == 'RTSTRUCT']
    for idx in tqdm(rtstructs):
        filename = df.iloc[idx].ReferencedFileID
        rtstruct = dicom.read_file(filename, stop_before_pixels=True)
        referenced_series_instance_uid = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
        df.at[idx, 'ReferencedSeriesInstanceUID'] = referenced_series_instance_uid
        df.loc[df.SeriesInstanceUID == referenced_series_instance_uid, 'CorrespondingRTSTRUCTReferencedFileID'] = filename

    del df['index']
    return df

def read_csvs(CACHEFILE, csv_dir):
    csvs = csv_dir.glob('*.csv')
    if not CACHEFILE.exists():
        print('making new dataframe...')
        df = read_dataframe(csvs)
        print('saving new dataframe to cache...')
        df.to_csv(CACHEFILE)
    else:
        print('reading dataframe from cache...')
        df = pd.read_csv(CACHEFILE, index_col=0, dtype={'PatientID':str, 'Rows': pd.Int64Dtype(), 'InstanceNumber': pd.Int64Dtype(), 'Columns': pd.Int64Dtype(), 'SeriesNumber': pd.Int64Dtype()})
    df['StudyDate'] = pd.to_datetime(df['StudyDate'], format='%Y-%m-%d')

    return df

def get_rtstruct_files(CACHEFILE, csv_dir):
    df = read_csvs(CACHEFILE, csv_dir)
    data = df.loc[df['Modality'] == 'CT', ('PatientID', 'CorrespondingRTSTRUCTReferencedFileID')]

    return data.drop_duplicates().dropna(subset=['CorrespondingRTSTRUCTReferencedFileID'])

def main():
    args = parse_args()
    CACHEFILE = Path('/home/pieter/git/thesis/cachefile.csv')
    data = get_rtstruct_files(CACHEFILE, args.csv_dir)

    npat = data.shape[0]
    for idx, (_, patient, rtstruct_file) in enumerate(data.itertuples()):
        print(f'patient {idx+1}/{npat}:', patient)
        folder_to_image = args.root_dir/patient

        if rtstruct_file:
            rtstruct_file = args.root_dir/rtstruct_file
        else:
            rtstruct_file = None

        structures = DicomRtstructReader(folder_to_image, rtstruct_file)
        write_to = args.write_to/f'{patient}'
        os.makedirs(write_to, exist_ok=True)
        image_filename = write_to/'image.nrrd'
        if not image_filename.exists():
            sitk.WriteImage(structures.image, image_filename, True)

        if rtstruct_file:
            for structure in ['LUNG_R', 'LUNG_L', 'LUNGs-GTVs', 'GTV']:

                i = 0
                while (write_to/(f"{structure.lower()}_" + str(i) + '.nrrd')).exists():
                    i += 1
                filebasename = filebasename + str(i) + '.nrrd'

                try:
                    sitk.WriteImage(structures.get_roi(structure), filebasename, True)
                except AssertionError as e:
                    print(e)
        print()


if __name__ == '__main__':
    main()