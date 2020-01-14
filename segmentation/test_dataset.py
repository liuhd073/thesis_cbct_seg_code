print(__name__)

from dataset import CervixDataset
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
    patients = os.listdir(args.root_dir)

    device = "cuda" # Run on GPU

    # Dataset for CBCTs
    image_shapes = pickle.load(open("CT_shapes.p", 'rb'))
    # cbct = patients[1] + "\\X01"
    # print(cbct)
    # image_shapes_0 = {cbct: image_shapes[cbct]}
    
    # Dataset for ONE CT image
    # image_shapes = pickle.load(open("CBCT_shapes.p", 'rb'))
    # cbct = patients[0] + "\\X01"
    # print(cbct)
    # image_shapes_0 = {cbct: image_shapes[cbct]}
        
    # ds = CervixDataset(args.root_dir, image_shapes_0, conebeams=True)
    ds = CervixDataset(args.root_dir, image_shapes, conebeams=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    for j in range(2):
        for i, (X, Y) in enumerate(dl):
            print("{} {}/{} X: {} Y: {}".format(j, i, len(dl), X.shape, Y.shape))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data", default="/data/cervix/patients", required=False)
    
    args = parser.parse_args()

    print(args)

    train(args)

