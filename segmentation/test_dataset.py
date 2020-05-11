"""
Author: Tessa Wagenaar
"""

from dataset import CTDataset
from dataset_CBCT import CBCTDataset
from dataset_combined import CombinedDataset
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
    img_arr = X[0, 0, middle_slice, :, :].detach().cpu()
    seg_arr_bladder = Y[:, 0, :, :, :].squeeze().detach().cpu()
    seg_arr_cervix = Y[:, 1, :, :, :].squeeze().detach().cpu()

    masked_img_bladder = np.array(
        plot_2d(img_arr, mask=seg_arr_bladder, mask_color="g", mask_threshold=0.5))
    masked_img_cervix = np.array(
        plot_2d(img_arr, mask=seg_arr_cervix, mask_color="g", mask_threshold=0.5))

    writer.add_image(
        f"{tag}/bladder_gt", Y[:, 0, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/cervix_gt", Y[:, 1, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/mask_bladder", masked_img_bladder, i, dataformats="HWC")
    writer.add_image(
        f"{tag}/mask_cervix", masked_img_cervix, i, dataformats="HWC")


def test(args):
    writer = SummaryWriter()
    files_CBCT = pickle.load(open("files_CBCT_clipped.p", 'rb'))
    files_CT = pickle.load(open("files_CT_clipped.p", 'rb'))
    transform_CBCT= transforms.Compose(
        [GaussianAdditiveNoise(0, 10), ClipAndNormalize(700, 1600)])#, RandomElastic((21,512,512))])
    ds_CBCT = CBCTDataset(files_CBCT, transform=transform_CBCT)
    transform_CT= transforms.Compose(
        [GaussianAdditiveNoise(0, 10), ClipAndNormalize(-100, 300)])#, RandomElastic((21,512,512))])
    ds_CT = CTDataset(files_CT, transform=transform_CT)
    ds_combined = CombinedDataset(ds_CT, ds_CBCT)
    dl = DataLoader(ds_combined, batch_size=1, shuffle=False, num_workers=12)
    
    seg_slices = 0
    no_seg_slices = 0

    for i, (X, Y) in enumerate(dl):
        logger.info("{}/{} X: {} Y: {}".format(i, len(ds_combined), X.shape, Y.shape))
        if len(Y.argmax(1).unique()) < 2:
            no_seg_slices += 1
        else:
            seg_slices += 1

        logger.debug("Unique:".format(Y.argmax(1).unique()))
        logger.debug(f"Min: {X.min()}, Max: {X.max()}")

        _log_images(X, Y, i, writer)
    print("Seg slices:", seg_slices)
    print("No seg slices:", no_seg_slices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Dataset')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix", required=False)

    args = parser.parse_args()

    test(args)
