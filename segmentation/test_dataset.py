"""
Author: Tessa Wagenaar
"""

from dataset import CTDataset
from dataset_CBCT import CBCTDataset
from dataset_combined import CombinedDataset
from dataset_duo import DuoDataset
from preprocess import *
from utils.plotting import plot_2d
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import pickle
import argparse
import torch
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def _log_images(X, Y, i, writer, tag="test_dataset"):
    middle_slice = X.shape[2] // 2
    img_arr = X[0, 0, middle_slice, :, :]
    seg_arr_bladder = Y[:, 0, :, :, :].squeeze()
    seg_arr_cervix = Y[:, 1, :, :, :].squeeze()
    seg_arr_other = Y[:, 2, :, :, :].squeeze()

    masked_img_bladder = np.array(
        plot_2d(img_arr, mask=seg_arr_bladder, mask_color="g", mask_threshold=0.5))
    masked_img_cervix = np.array(
        plot_2d(img_arr, mask=seg_arr_cervix, mask_color="g", mask_threshold=0.5))

    logger.debug(f"{seg_arr_other.shape}, {seg_arr_other.sum()}")

    # print(Y[:, 0, :, :, :].squeeze().unique())
    writer.add_image(
        f"{tag}/bladder_gt", seg_arr_bladder, i, dataformats="HW")
    writer.add_image(
        f"{tag}/cervix_gt", seg_arr_cervix, i, dataformats="HW")
    # writer.add_image(
        # f"{tag}/other_gt", seg_arr_other, i, dataformats="HW")
    writer.add_image(
        f"{tag}/mask_bladder", masked_img_bladder, i, dataformats="HWC")
    writer.add_image(
        f"{tag}/mask_cervix", masked_img_cervix, i, dataformats="HWC")


def test(args):
    writer = SummaryWriter()
    # Load datasets
    files_CBCT_train = pickle.load(
        open("files_CBCT_train.p", 'rb'))  # training
    files_CBCT_val = pickle.load(
        open("files_CBCT_validation.p", 'rb'))  # validation
    files_CBCT_test = pickle.load(open("files_CBCT_test.p", 'rb'))

    # Load datasets
    files_CT_train = pickle.load(
        open("files_CT_train.p", 'rb'))  # training
    files_CT_val = pickle.load(
        open("files_CT_validation.p", 'rb'))  # validation

    # Load datasets
    files_CBCT_CT_train = pickle.load(
        open("files_CBCT_CT_train.p", 'rb'))  # training
    files_CBCT_CT_val = pickle.load(
        open("files_CBCT_CT_validation.p", 'rb'))  # validation

    transform_CBCT= transforms.Compose(
        [GaussianAdditiveNoise(0, 10), RandomElastic((21,512,512)), ClipAndNormalize(800, 1250)])#, RandomElastic((21,512,512))])
    transform_CT= transforms.Compose(
        [GaussianAdditiveNoise(0, 10), RandomElastic((21,512,512)), ClipAndNormalize(800, 1250)])#, RandomElastic((21,512,512))])
    ds_CT = CTDataset(files_CT_val, transform=transform_CT)
    ds_CBCT = CBCTDataset(files_CBCT_val, transform=transform_CBCT)
    ds_combined = CombinedDataset(ds_CT, ds_CBCT)
    dl = DataLoader(ds_CBCT, batch_size=1, shuffle=False, num_workers=12)

    seg_slices = 0
    no_seg_slices = 0
    j=0
    n = 0
    patient = files_CBCT_val[n][0]
    shape = files_CBCT_val[n][1][0]

    for i, (X, Y) in enumerate(dl):
        logger.info("{}/{} X: {} Y: {}".format(i, len(ds_CBCT), X.shape, Y.shape))
        if len(Y.argmax(1).unique()) < 2:
            no_seg_slices += 1
        else:
            seg_slices += 1

        logger.debug("Unique:".format(Y.argmax(1).unique()))
        logger.debug(f"Min: {X.min()}, Max: {X.max()}")

        _log_images(X.numpy(), Y.numpy(), i, writer, tag=patient)
        j+=1
        if j >= shape:
            n += 1
            patient = files_CBCT_val[n][0]
            if patient == "CT":
                patient = files_CBCT_val[n][2].parent.stem
            shape = files_CBCT_val[n][1][0]
            j=0
    print("Seg slices:", seg_slices)
    print("No seg slices:", no_seg_slices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Dataset')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix", required=False)

    args = parser.parse_args()

    test(args)
