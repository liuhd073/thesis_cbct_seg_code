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

def _log_images(X_cbct, X_ct, Y_cbct, Y_ct, i, writer, tag="test_dataset"):
    middle_slice = X_ct.shape[2] // 2
    img_arr = X_cbct[0, 0, middle_slice, :, :].detach().cpu()
    seg_arr_bladder = Y_cbct[:, 0, :, :, :].squeeze().detach().cpu()
    seg_arr_cervix = Y_cbct[:, 1, :, :, :].squeeze().detach().cpu()

    masked_img_bladder = np.array(
        plot_2d(img_arr, mask=seg_arr_bladder, mask_color="g", mask_threshold=0.5))
    masked_img_cervix = np.array(
        plot_2d(img_arr, mask=seg_arr_cervix, mask_color="g", mask_threshold=0.5))

    writer.add_image(
        f"{tag}/mask_bladder_CBCT", masked_img_bladder, i, dataformats="HWC")
    writer.add_image(
        f"{tag}/mask_cervix_CBCT", masked_img_cervix, i, dataformats="HWC")

    img_arr = X_ct[0, 0, middle_slice, :, :].detach().cpu()
    seg_arr_bladder = Y_ct[:, 0, :, :, :].squeeze().detach().cpu()
    seg_arr_cervix = Y_ct[:, 1, :, :, :].squeeze().detach().cpu()

    masked_img_bladder = np.array(
        plot_2d(img_arr, mask=seg_arr_bladder, mask_color="g", mask_threshold=0.5))
    masked_img_cervix = np.array(
        plot_2d(img_arr, mask=seg_arr_cervix, mask_color="g", mask_threshold=0.5))

    writer.add_image(
        f"{tag}/mask_bladder_CT", masked_img_bladder, i, dataformats="HWC")
    writer.add_image(
        f"{tag}/mask_cervix_CT", masked_img_cervix, i, dataformats="HWC")


def test():
    writer = SummaryWriter()

    # Load datasets
    files_CBCT_CT_train = pickle.load(
        open("files_CBCT_CT_train.p", 'rb'))  # training
    files_CBCT_CT_val = pickle.load(
        open("files_CBCT_CT_validation.p", 'rb'))  # validation

    transform_CT= transforms.Compose(
        [GaussianAdditiveNoise(0, 10), RandomElastic((21,512,512)), ClipAndNormalize(800, 1250)])#, RandomElastic((21,512,512))])
    ds_duo = DuoDataset(files_CBCT_CT_val, transform=transform_CT, return_patient=True)
    dl = DataLoader(ds_duo, batch_size=1, shuffle=False, num_workers=10)

    seg_slices = 0
    no_seg_slices = 0

    for i, (patient, X_cbct, X_ct, Y_cbct, Y_ct) in enumerate(dl):
        logger.info("{}/{} X: {} Y: {}".format(i, len(ds_duo), X_cbct.shape, Y_cbct.shape))
        if len(Y_cbct.argmax(1).unique()) < 2:
            no_seg_slices += 1
        else:
            seg_slices += 1

        logger.debug(f"Unique: {Y_cbct.argmax(1).unique()} {Y_ct.argmax(1).unique()}")
        logger.debug(f"Min: {X_cbct.min()}, {X_ct.min()}, Max: {X_cbct.max()}, {X_ct.max()}")

        _log_images(X_cbct, X_ct, Y_cbct, Y_ct, i, writer, tag=patient)
        # _log_images(X_ct, Y_ct, i, writer, tag="Duo CT")

    print("Seg slices:", seg_slices)
    print("No seg slices:", no_seg_slices)


if __name__ == "__main__":
    test()
