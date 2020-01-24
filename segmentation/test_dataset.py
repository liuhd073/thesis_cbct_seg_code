# from dataset import CervixDataset
from dataset_extra_CTs import CervixDataset
from torch.utils.data import DataLoader
from resblockunet import ResBlockUnet
from torch.optim import Adam

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import argparse
import pickle
import torch
import os


def train(args):
    image_shapes = pickle.load(open("extra_CT_shapes.p", 'rb'))
    ds = CervixDataset(args.root_dir, image_shapes)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    for j in range(2):
        for i, (X, Y) in enumerate(dl):
            print("{} {}/{} X: {} Y: {}".format(j, i, len(dl), X.shape, Y.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix/patients", required=False)

    args = parser.parse_args()

    print(args)

    train(args)
