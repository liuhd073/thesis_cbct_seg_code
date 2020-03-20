import pickle
import os
import torch
import argparse
import neptune
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
from preprocess import Clip, NormalizeHV, GaussianAdditiveNoise
from utils.plotting import plot_2d
from mini_model import UNetResBlocks  # , UNet
import time
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_loss_func(loss_func="BCE"):
    if loss_func == "BCE":
        return nn.BCEWithLogitsLoss()
    if loss_func == "NLL":
        weights = torch.Tensor([1, 1, 0.1]).to("cuda")
        if args.topk < 1.0:
            return nn.NLLLoss(weight=weights, reduction="none")
        else:
            return nn.NLLLoss(weight=weights)


def get_model(args, device):
    logger.info("Loading Model...")
    model = ResBlockUnet(1, 3, (1, 512, 512), use_group_norm=args.use_group_norm)
    if args.load_model:
        model.load_state_dict(torch.load(open(args.model_file, 'rb')))
        logger.info("Model loaded!")
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
    middle_slice = X.shape[2] // 2
    img_arr = X[0, 0, middle_slice, :, :].detach().cpu()
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
        # torch.cuda.empty_cache()
        if args.mc_train:
            if len(Y.argmax(1).unique()) < 2:
                continue

        if args.equal_train:
            if len(Y.argmax(1).unique()) < 2 and np.random.rand(1) > 0.25:
                continue
        torch.cuda.empty_cache()
        Y_hat = model(X)
        if args.loss_func == "NLL":
            Y_hat = softmax(Y_hat)
        loss = criterion(Y_hat, Y.argmax(1)).mean()
        
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

    for i, (X, Y) in enumerate(dl):
        model.train()
        X, Y = X.to(device).float(), Y.to(device).float()

        if args.mc_train:
            if len(Y.argmax(1).unique()) < 2:
                continue
        if args.equal_train:
            if len(Y.argmax(1).unique()) < 2 and np.random.rand(1) > 0.25:
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
        tmp_losses.append(losses["train"][-1])
        writer.add_scalar("loss/train", losses["train"][-1], true_i)
        neptune.send_metric("loss/train", true_i, losses["train"][-1])

        if true_i % args.eval_every == 0:
            eval_loss = evaluate(dl_val, writer, model, device, criterion, j, max_iters=250)
            writer.add_scalar("loss/validation", eval_loss, true_i)
            neptune.send_metric("loss/validation", true_i, eval_loss)
            logger.info("Iteration: {}/{} Validation Loss: {}".format(true_i,
                                                            args.max_iters, eval_loss))
            if eval_loss < losses["best"]:
                losses["best"] = eval_loss
                model_fn = args.model_dir / "best_model_{}.pt".format(args.loss_func)
                torch.save(model.state_dict(), model_fn)
                logger.info("Best model saved: best_model_{}.pt".format(args.loss_func))
            losses["validation"].append(eval_loss)

        if true_i % args.print_every == 0:
            logger.info("True Iteration: {} Epoch: {}/{} Iteration: {}/{} Loss: {}".format(true_i, j,
                                                                  args.max_epochs, i, len(dl), sum(tmp_losses)/len(tmp_losses)))
            tmp_losses = []
            if args.save_imgs:
                _log_images(X, Y, Y_hat, true_i, writer, "train")

        if true_i % args.save_every == 0:
            # Save Model
            model_fn = args.model_dir / "model_{}_{}.pt".format(true_i, args.loss_func)
            torch.save(model.state_dict(), model_fn)
            logger.info("Model saved in model_{}_{}.pt".format(true_i, args.loss_func))
            if true_i >= args.max_iters:
                exit(1)

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

    transform = transforms.Compose([GaussianAdditiveNoise(0, 0.01), Clip(), NormalizeHV()])
    ds = CTDataset(files_train, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=args.shuffle, num_workers=12)

    ds_val = CTDataset(files_val, transform=transform)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=args.shuffle, num_workers=12)

    writer = SummaryWriter()

    # model = UNetResBlocks().to(device)
    model = get_model(args, device)
    logger.info("Model Loaded")
    optimizer = Adam(model.parameters(), lr=args.lr)
    # TODO: Add LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

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

    logger.info("Start Training...")
    true_i = 1
    for j in range(args.max_epochs):
        losses, true_i = train(model, dl, dl_val, optimizer,
                               criterion, args, writer, device, j, true_i, losses)

        scheduler.step()
        loss_fn = args.loss_dir / "losses.p"
        pickle.dump(losses["train"], open(loss_fn, 'wb'))
        loss_fn = args.loss_dir / "losses_val.p"
        pickle.dump(losses["validation"], open(loss_fn, 'wb'))

    logger.info("End training, save final model...")
    writer.flush()
    torch.save(model.state_dict(), "final_model.pt")


def parse_args():
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-experiment_name", help="Set name of the experiment in Neptune",
                        default="Experiment", required=False)
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

    parser.add_argument("-max_epochs", help="Maximum number of iterations",
                        default=100, required=False, type=int)
    parser.add_argument("-max_iters", help="Maximum number of iterations",
                        default=150000, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations",
                        default=10, required=False, type=int)
    parser.add_argument("-save_every", help="Save model every X epochs",
                        default=1, required=False, type=int)
    parser.add_argument("-eval_every", help="Evaluate model every X epochs using validation set",
                        default=1, required=False, type=int)

    args = parser.parse_args()

    args.results_folder = Path(args.results_folder)

    return args


def get_params(args):
    PARAMS = {}
    PARAMS["Model"] = "3D U-Net + ResNet"
    PARAMS["shuffle"] = args.shuffle
    PARAMS["Equal_train"] = args.equal_train
    PARAMS["Top_k"] = args.topk
    PARAMS["Save_every"] = args.save_every
    PARAMS["Eval_every"] = args.eval_every
    PARAMS["Learning_Rate"] = args.lr
    PARAMS["Pooling"] = "avg"
    return PARAMS


if __name__ == "__main__":
    args = parse_args()

    PARAMS = get_params(args)
    neptune.init('twagenaar/Thesis-CT-seg')
    neptune.create_experiment(name=args.experiment_name, params=PARAMS)
    neptune.append_tag("running")
    main(args)

    neptune.remove_tag("running")
    neptune.append_tag("finished")
    neptune.stop()