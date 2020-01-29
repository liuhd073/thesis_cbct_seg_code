from dataset import CervixDataset
from dataset_extra_CTs import ExtraCervixDataset
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
    writer = SummaryWriter()
    image_shapes = pickle.load(open("CT_shapes_validation.p", 'rb'))
    transform = transforms.Compose([Clip(), NormalizeHV()])
    ds = CervixDataset(args.root_dir, image_shapes, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    for j in range(2):
        for i, (X, Y) in enumerate(dl):
            print("{} {}/{} X: {} Y: {}".format(j, i, len(dl), X.shape, Y.shape))
            # writer.add_image(
            #     "images_true/0", Y[:, 0, :, :, :].squeeze(), i, dataformats="HW")
            # writer.add_image(
            #     "images_true/1", Y[:, 1, :, :, :].squeeze(), i, dataformats="HW")
            # writer.add_image(
            #     "images_true/2", Y[:, 2, :, :, :].squeeze(), i, dataformats="HW")
            # writer.add_image( 
            #     "images_true/X", X[:, :, 10:11, :, :].squeeze(), i, dataformats="HW")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix/patients", required=False)

    args = parser.parse_args()

    print(args)

    train(args)
