"""
Author: Tessa Wagenaar
"""

from dataset import CTDataset
from preprocess import Clip, NormalizeHV, NormalizeIMG
from torch.utils.data import DataLoader
from resblockunet import ResBlockUnet
from torch.optim import Adam

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

import argparse
import pickle
import torch
import os


def train(args):
    j=0
    writer = SummaryWriter()
    files_train = pickle.load(open("files_test.p", 'rb'))
    transform = transforms.Compose([Clip(), NormalizeHV()])
    ds = CTDataset(files_train, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    seg_slices = 0
    no_seg_slices = 0

    for i, (X, Y) in enumerate(dl):
        print("{} {}/{} X: {} Y: {}".format(j, i, len(dl), X.shape, Y.shape))
        if len(Y.argmax(1).unique()) < 2:
            no_seg_slices += 1
        else: 
            seg_slices += 1
    
        print(X.min(), X.max())
        writer.add_image(
            "dataset_test/bladder", Y[:, 0, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "dataset_test/cervix_uterus", Y[:, 1, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image( 
            "dataset_test/X", X[:, :, 10:11, :, :].squeeze(), i, dataformats="HW")

    print("Seg slices:", seg_slices)
    print("No seg slices:", no_seg_slices)


    # image_shapes = pickle.load(open("extra_CT_shapes_validation.p", 'rb'))
    # ds = ExtraCervixDataset("/data/cervix/converted", image_shapes, transform=transform)
    # dl = DataLoader(ds, batch_size=1, shuffle=False)

    # for i, (X, Y) in enumerate(dl):
    #     print("{} {}/{} X: {} Y: {}".format(j, i, len(dl), X.shape, Y.shape))
    #     # writer.add_image(
    #     #     "images_true/0", Y[:, 0, :, :, :].squeeze(), i, dataformats="HW")
    #     # writer.add_image(
    #     #     "images_true/1", Y[:, 1, :, :, :].squeeze(), i, dataformats="HW")
    #     # writer.add_image(
    #     #     "images_true/2", Y[:, 2, :, :, :].squeeze(), i, dataformats="HW")
    #     # writer.add_image( 
    #     #     "images_true/X", X[:, :, 10:11, :, :].squeeze(), i, dataformats="HW")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix/patients", required=False)

    args = parser.parse_args()

    print(args)

    train(args)
