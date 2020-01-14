print(__name__)

from dataset import CervixDataset
from preprocess import Clip, NormalizeHV
from torch.utils.data import DataLoader
from resblockunet import ResBlockUnet
from torch.optim import Adam

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import argparse
import pickle 
import torch
import os

def get_model(args):
    print("Loading Model...")
    model = ResBlockUnet(1, 3, (1, 512, 512))
    if args.load_model:
        model.load_state_dict(torch.load(open(args.model_file, 'rb')))
        print("Model loaded!")
    model.to(device)
    
    return model


def evaluate(dl_val, model):
    dataset_mean = 0.7
    model.eval()
    for (X,Y) in dl_val:
        Y_hat = model(X)

    return loss


def train(model, dl, optimizer, criterion, softmax, args, writer, device, j):
    dataset_mean = 0.7
    losses = []
    tmp_losses = []
    for i, (X, Y) in enumerate(dl):
        model.train()
        X, Y = X.to(device).float(), Y.to(device).float()
        X = X - dataset_mean

        if args.mc_train:
            if len(Y.argmax(1).unique()) < 2:
                continue

        Y_hat = model(X)
        assert Y_hat.shape == Y.shape, "output and classification must be same shape"
        loss = criterion(Y_hat, Y)
        
        Y_hat = softmax(Y_hat)

        losses.append(loss.item())
        tmp_losses.append(loss.item())
        writer.add_scalar("loss/train", loss.item(), j * len(dl) + i)

        torch.cuda.empty_cache()

        if (j * len(dl) + i) % args.print_every == 0:
            print("Iteration: {}/{} Loss: {}".format(j * len(dl) + i, len(dl) * args.max_iters, loss.item()))

        # Train model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum(tmp_losses) / len(tmp_losses)

    return avg_loss, losses


def main(args):
    patients = os.listdir(args.root_dir)

    device = "cuda" # Run on GPU

    image_shapes = pickle.load(open("CT_shapes.p", 'rb')) # change to train
        
    transform = transforms.Compose([Clip(), NormalizeHV()])
    ds = CervixDataset(args.root_dir, image_shapes, transform=transform, conebeams=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    writer = SummaryWriter()

    model = get_model(args)
    optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = nn.BCEWithLogitsLoss()
    softmax = nn.Sigmoid()

    best_loss = float('inf')
    image_losses = []

    print("Start Training...")

    for j in range(args.max_iters):
        avg_loss, losses = train(model, dl, optimizer, criterion, softmax, args, writer, device, j)
        image_losses.append(avg_loss)

        writer.add_scalar("loss/image", avg_loss, j)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved")

        print("Iteration: {}/{} Average Loss: {}".format((j + 1) * len(dl), len(dl) * args.max_iters, avg_loss))

        if j % args.save_every == 0:
            # Save Model
            torch.save(model.state_dict(), "model_{}.pt".format(j))

        if j == 75:
            args.mc_train = False

        # TODO: COMBINE LOSSES

     
    print("End training, save final model...")
    torch.save(model.state_dict(), "final_model.pt")
    pickle.dump(losses, open("losses.p", 'wb'))
    pickle.dump(image_losses, open("avg_losses.p", 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data", default="/data/cervix/patients", required=False)
    parser.add_argument("-model_file", help="Get the file containing the model weights", default="final_model.pt", required=False)
    parser.add_argument("-load_model", help="Get the model weights", default=False, required=False, action="store_true")

    parser.add_argument("-lr", help="Learning Rate", default=0.0001, required=False, type=float)
    parser.add_argument("-mc_train", help="Only train using images with at least 2 classes", default=False, required=False, action="store_true")

    parser.add_argument("-max_iters", help="Maximum number of iterations", default=10, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations", default=10, required=False, type=int)
    parser.add_argument("-save_every", help="Save model every X epochs", default=1, required=False, type=int)
    
    args = parser.parse_args()

    print(args)

    main(args)

# python train.py -max_iters 100 -save_every 10 -mc_train
# Train on all 10 patient CT images. 