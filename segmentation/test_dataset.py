"""
Author: Tessa Wagenaar
"""

from dataset import CTDataset
from dataset_CBCT import CBCTDataset
from preprocess import *
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


def test(args):
    writer = SummaryWriter()
    files_train = pickle.load(open("files_CBCT.p", 'rb'))
    # transform = transforms.Compose(
    #     [GaussianAdditiveNoise(0, 20), Clip(), NormalizeHV()])
    transform = transforms.Compose(
        [NormalizeIMG()])
    ds = CBCTDataset(files_train, transform=transform)
    # dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=6)
    seg_slices = 0
    no_seg_slices = 0

    for i in range(10, len(ds), 100):
    # for i, (X, Y) in enumerate(dl):
        X, Y = ds.__getitem__(i)
        X, Y = torch.tensor(X).unsqueeze(0), torch.tensor(Y).unsqueeze(0)
        logger.info("{}/{} X: {} Y: {}".format(i, len(ds), X.shape, Y.shape))
        if len(Y.argmax(1).unique()) < 2:
            no_seg_slices += 1
        else:
            seg_slices += 1

        logger.debug("Unique:".format(Y.argmax(1).unique()))

        writer.add_image(
            "dataset_test/bladder", Y[:, 0, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "dataset_test/cervix_uterus", Y[:, 1, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "dataset_test/other", Y[:, 2, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "dataset_test/X", X[:, :, X.shape[2]//2, :, :].squeeze(), i, dataformats="HW")

    print("Seg slices:", seg_slices)
    print("No seg slices:", no_seg_slices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Dataset')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix", required=False)

    args = parser.parse_args()

    print(args)

    test(args)
