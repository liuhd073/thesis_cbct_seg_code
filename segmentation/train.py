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
from tqdm import tqdm
from pathlib import Path
from resblockunet import ResBlockUnet
from torch.utils.data import DataLoader
from dataset import CTDataset
from preprocess import Clip, NormalizeHV
from utils.plotting import plot_2d
from mini_model import UNetResBlocks  # , UNet
import time


def get_loss_func(loss_func="BCE"):
    if loss_func == "BCE":
        return nn.BCEWithLogitsLoss()
    if loss_func == "NLL":
        weights = torch.Tensor([1, 1, 0.1]).to("cuda")
        return nn.NLLLoss(weight=weights)  # , reduction="none"


def get_model(args, device):
    print("Loading Model...")
    model = ResBlockUnet(1, 3, (1, 512, 512), use_group_norm=args.use_group_norm)
    if args.load_model:
        model.load_state_dict(torch.load(open(args.model_file, 'rb')))
        print("Model loaded!")
    model.to(device)

    return model


def check_results_folder(args):
    # res_dir = Path(args.results_folder)
    res_dir = args.results_folder
    if not res_dir.exists():
        res_dir.mkdir()

    model_dir = res_dir / "models"
    if not model_dir.exists():
        model_dir.mkdir()

    loss_dir = res_dir / "losses"
    if not loss_dir.exists():
        loss_dir.mkdir()

    args.model_dir = model_dir
    args.loss_dir = loss_dir


def _log_images(X, Y, Y_hat, i, writer, tag="train"):
    img_arr = X[0, 0, 3, :, :].detach().cpu()
    seg_arr_bladder = Y[:, 0, :, :, :].squeeze().detach().cpu()
    seg_arr_cervix = Y[:, 1, :, :, :].squeeze().detach().cpu()

    out_arr_bladder = Y_hat.exp()[:, 0, :, :, :].squeeze().detach().cpu()
    out_arr_cervix = Y_hat.exp()[:, 1, :, :, :].squeeze().detach().cpu()

    masked_img_bladder = np.array(
        plot_2d(img_arr, mask=out_arr_bladder, mask_color="r", mask_threshold=0.5))
    masked_img_cervix = np.array(
        plot_2d(img_arr, mask=out_arr_cervix, mask_color="r", mask_threshold=0.5))

    masked_img_bladder = torch.from_numpy(
        np.array(plot_2d(masked_img_bladder, mask=seg_arr_bladder, mask_color="g")))
    masked_img_cervix = torch.from_numpy(
        np.array(plot_2d(masked_img_cervix, mask=seg_arr_cervix, mask_color="g")))

    writer.add_image(
        f"{tag}/bladder", Y_hat.exp()[:, 0, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/cervix", Y_hat.exp()[:, 1, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/bladder_gt", Y[:, 0, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/cervix_gt", Y[:, 1, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/mask_bladder", masked_img_bladder, i, dataformats="HWC")
    writer.add_image(
        f"{tag}/mask_cervix", masked_img_cervix, i, dataformats="HWC")


def evaluate(dl_val, writer, model, device, criterion, j, max_iters=None):
    dataset_mean = 0.1
    losses = []
    softmax = nn.LogSoftmax(1)
    i = 1

    model.eval()
    for X, Y in tqdm(dl_val):
        if max_iters is not None and i > max_iters:
            break
        X, Y = X.to(device).float(), Y.to(device).float()
        X = X - dataset_mean
        torch.cuda.empty_cache()
        if args.mc_train:
            if len(Y.argmax(1).unique()) < 2:
                continue

        if args.equal_train:
            if len(Y.argmax(1).unique()) < 2 and np.random.rand(1) > 0.25:
                continue
        Y_hat = model(X)
        if args.loss_func == "NLL":
            Y_hat = softmax(Y_hat)
        loss = criterion(Y_hat, Y.argmax(1))
        topkloss, _ = torch.topk(loss.flatten(), int(args.topk*loss.numel()))
        loss = topkloss.mean()

        if args.save_imgs:
            _log_images(X, Y, Y_hat, i, writer, "validation")

        losses.append(loss.detach().cpu().item())
        torch.cuda.empty_cache()
        i += 1

    loss = np.mean(losses)
    return loss


def train(model, dl, dl_val, optimizer, criterion, args, writer, device, j, true_i, losses):
    dataset_mean = 0.1
    softmax = nn.LogSoftmax(1)
    # true_i = len(dl) * j + 1
    tmp_losses = []
    best_loss = float("inf")

    for (X, Y) in dl:
        model.train()
        X, Y = X.to(device).float(), Y.to(device).float()

        if args.mc_train:
            if len(Y.argmax(1).unique()) < 2:
                continue

        Y_hat = model(X - dataset_mean)
        assert Y_hat.shape == Y.shape, "output and classification must be same shape {} {}".format(
            Y.shape, Y_hat.shape)

        if args.loss_func == "NLL":
            Y_hat = softmax(Y_hat)
        loss = criterion(Y_hat, Y.argmax(1))
        topkloss, _ = torch.topk(loss.flatten(), int(args.topk*loss.numel()))
        loss = topkloss.mean()
        
        # Train model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        losses["train"].append(loss.detach().cpu().item())
        tmp_losses.append(losses[-1])
        writer.add_scalar("loss/train", losses[-1], true_i)

        if args.save_imgs:
            _log_images(X, Y, Y_hat, true_i, writer, "train")

        if true_i % args.eval_every == 0:
            eval_loss = evaluate(dl_val, writer, model, device, criterion, j)
            writer.add_scalar("loss/validation", eval_loss, true_i)
            print("Epoch: {}/{} Validation Loss: {}".format(true_i,
                                                            len(dl) * args.max_iters, eval_loss))
            if eval_loss < losses["best"]:
                losses["best"] = eval_loss
                model_fn = args.model_dir / "best_model_{}.pt".format(args.loss_func)
                torch.save(model.state_dict(), model_fn)
                print("Best model saved: best_model_{}.pt".format(args.loss_func))
            losses["validation"].append(eval_loss)

        if true_i % args.print_every == 0:
            print("Epoch: {}/{} Iteration: {}/{} Loss: {}".format(j,
                                                                  args.max_iters, true_i, len(dl), sum(tmp_losses)/len(tmp_losses)))
            tmp_losses = []

        if true_i % args.save_every == 0:
            # Save Model
            model_fn = args.model_dir / "model_{}_{}.pt".format(true_i, args.loss_func)
            torch.save(model.state_dict(), model_fn)
            print("Model saved in model_{}_{}.pt".format(true_i, args.loss_func))

        true_i += 1

    return losses, true_i


def main(args):
    check_results_folder(args)
    device = "cuda"  # Run on GPU

    # Load datasets
    files_train = pickle.load(
        open("files_train.p", 'rb'))  # training
    files_val = pickle.load(
        open("files_validation.p", 'rb'))  # validation

    transform = transforms.Compose([Clip(), NormalizeHV()])
    ds = CTDataset(files_train, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=args.shuffle)

    ds_val = CTDataset(files_val, transform=transform)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=args.shuffle)

    writer = SummaryWriter()

    model = UNetResBlocks().to(device)
    # model = get_model(args, device)
    print("Model Loaded")
    optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = get_loss_func(args.loss_func)

    all_losses = []
    eval_losses = []
    best_loss = float("inf")

    if args.load_losses:
        loss_fn = args.results_folder / "losses" / "losses.p"
        all_losses = pickle.load(open(loss_fn, 'rb'))
        loss_fn = args.results_folder / "losses" / "losses_val.p"
        eval_losses = pickle.load(open(loss_fn, 'rb'))
        best_loss = np.min(eval_losses)
        
    losses = {"train": all_losses, "validation": eval_losses, "best": best_loss}

    print("Start Training...")
    true_i = 1
    for j in range(args.max_iters):
        losses, true_i = train(model, dl, dl_val, optimizer,
                               criterion, args, writer, device, j, true_i, losses)

        loss_fn = args.loss_dir / "losses.p"
        pickle.dump(losses["train"], open(loss_fn, 'wb'))
        loss_fn = args.loss_dir / "losses_val.p"
        pickle.dump(losses["validation"], open(loss_fn, 'wb'))

    print("End training, save final model...")
    writer.flush()
    torch.save(model.state_dict(), "final_model.pt")


def parse_args():
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix", required=False)
    parser.add_argument("-model_file", help="Get the file containing the model weights",
                        default="final_model.pt", required=False)
    parser.add_argument("-load_model", help="Get the model weights",
                        default=False, required=False, action="store_true")
    parser.add_argument("-load_losses", help="Get the model weights",
                        default=False, required=False, action="store_true")
    parser.add_argument("-save_imgs", help="Save the training images to tensorboard",
                        default=False, required=False, action="store_true")
    parser.add_argument("-results_folder", help="Get the folder to store the results",
                        default="/app/results", required=False)

    parser.add_argument("-lr", help="Learning Rate",
                        default=0.0001, required=False, type=float)
    parser.add_argument("-topk", help="Top K Loss",
                        default=1.0, required=False, type=float)
    parser.add_argument("-loss_func", help="Loss Function: BCE/NLL",
                        default="BCE", required=False, type=str)
    parser.add_argument("-mc_train", help="Only train using images with at least 2 classes",
                        default=False, required=False, action="store_true")
    parser.add_argument("-equal_train", help="Only train using images with at least 2 classes",
                        default=False, required=False, action="store_true")
    parser.add_argument("-use_group_norm", help="Implement group norm in model",
                        default=False, required=False, action="store_true")
    parser.add_argument("-no_shuffle", dest="shuffle", help="Shuffle dataset",
                        default=True, required=False, action="store_false")

    parser.add_argument("-max_iters", help="Maximum number of iterations",
                        default=10, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations",
                        default=10, required=False, type=int)
    parser.add_argument("-save_every", help="Save model every X epochs",
                        default=1, required=False, type=int)
    parser.add_argument("-eval_every", help="Evaluate model every X epochs using validation set",
                        default=1, required=False, type=int)

    args = parser.parse_args()

    args.results_folder = Path(args.results_folder)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
