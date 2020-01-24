import pickle
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from torch.optim import Adam
from resblockunet import ResBlockUnet
from torch.utils.data import DataLoader
from preprocess import Clip, NormalizeHV
from dataset_extra_CTs import CervixDataset
print(__name__)

# from dataset import CervixDataset


def get_loss_func(loss_func="BCE"):
    if loss_func == "BCE":
        return nn.BCEWithLogitsLoss()
    if loss_func == "NLL":
        weights = torch.Tensor([1, 1, 0.0001]).to("cuda")
        return nn.NLLLoss(weight=weights)


def get_model(args, device):
    print("Loading Model...")
    model = ResBlockUnet(1, 3, (1, 512, 512))
    if args.load_model:
        model.load_state_dict(torch.load(open(args.model_file, 'rb')))
        print("Model loaded!")
    model.to(device)

    return model


def evaluate(dl_val, model, device, criterion, writer, j):
    dataset_mean = 0.7
    losses = []
    model.eval()

    for (X, Y) in dl_val:
        X, Y = X.to(device).float(), Y.to(device).float()
        X = X - dataset_mean
        torch.cuda.empty_cache()
        Y_hat = model(X)
        loss = criterion(Y_hat, Y)

        losses.append(loss.item())
        torch.cuda.empty_cache()

    loss = np.mean(losses)
    writer.add_scalar("loss/validation", loss, j)
    return loss


def train(model, dl, optimizer, criterion, args, writer, device, j):
    dataset_mean = 0.7
    losses = []
    tmp_losses = []
    softmax = nn.LogSoftmax(1)
    for i, (X, Y) in enumerate(dl):
        model.train()
        X, Y = X.to(device).float(), Y.to(device).float()
        X = X - dataset_mean

        if args.mc_train:
            if len(Y.argmax(1).unique()) < 2:
                continue

        Y_hat = model(X)
        assert Y_hat.shape == Y.shape, "output and classification must be same shape"
        if args.loss_func == "NLL":
            Y_hat = softmax(Y_hat)
            Y = Y.argmax(1)
        loss = criterion(Y_hat, Y)

        losses.append(loss.item())
        tmp_losses.append(loss.item())
        writer.add_scalar("loss/train", loss.item(), j * len(dl) + i)

        torch.cuda.empty_cache()

        if (j * len(dl) + i) % args.print_every == 0:
            print("Iteration: {}/{} Loss: {}".format(j * len(dl) +
                                                     i, len(dl) * args.max_iters, loss.item()))

        # Train model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum(tmp_losses) / len(tmp_losses)

    return avg_loss, losses


def main(args):
    device = "cuda"  # Run on GPU

    image_shapes = pickle.load(
        open("extra_CT_shapes_train.p", 'rb'))  # training
    image_shapes_val = pickle.load(
        open("extra_CT_shapes_validation.p", 'rb'))  # validation

    transform = transforms.Compose([Clip(), NormalizeHV()])
    ds = CervixDataset(args.root_dir, image_shapes, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    ds_val = CervixDataset(
        args.root_dir, image_shapes_val, transform=transform)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

    writer = SummaryWriter()

    model = get_model(args, device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = get_loss_func(args.loss_func)

    best_loss = float('inf')
    image_losses = []
    all_losses = []

    print("Start Training...")

    for j in range(args.max_iters):
        avg_loss, losses = train(model, dl, optimizer,
                                 criterion, args, writer, device, j)
        image_losses.append(avg_loss)
        all_losses += losses

        writer.add_scalar("loss/image", avg_loss, j)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved")

        print("Iteration: {}/{} Average Loss: {}".format((j + 1)
                                                         * len(dl), len(dl) * args.max_iters, avg_loss))

        if j % args.save_every == 0:
            # Save Model
            torch.save(model.state_dict(), "model_{}.pt".format(j))
            print("Model saved in model_{}.pt".format(j))

        if j % args.eval_every == 0:
            eval_loss = evaluate(dl_val, model, device, criterion, writer, j)
            print("Epoch: {}/{} Validation Loss: {}".format(j,
                                                            args.max_iters, eval_loss))

    print("End training, save final model...")
    torch.save(model.state_dict(), "final_model.pt")
    pickle.dump(all_losses, open("losses.p", 'wb'))
    pickle.dump(image_losses, open("avg_losses.p", 'wb'))


def parse_args():
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix/patients", required=False)
    parser.add_argument("-model_file", help="Get the file containing the model weights",
                        default="final_model.pt", required=False)
    parser.add_argument("-load_model", help="Get the model weights",
                        default=False, required=False, action="store_true")

    parser.add_argument("-lr", help="Learning Rate",
                        default=0.0001, required=False, type=float)
    parser.add_argument("-loss_func", help="Loss Function: BCE/NLL",
                        default="BCE", required=False, type=str)
    parser.add_argument("-mc_train", help="Only train using images with at least 2 classes",
                        default=False, required=False, action="store_true")

    parser.add_argument("-max_iters", help="Maximum number of iterations",
                        default=10, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations",
                        default=10, required=False, type=int)
    parser.add_argument("-save_every", help="Save model every X epochs",
                        default=1, required=False, type=int)
    parser.add_argument("-eval_every", help="Evaluate model every X epochs using validation set",
                        default=1, required=False, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python train.py -max_iters 100 -save_every 10 -mc_train
# Train on all 10 patient CT images.
