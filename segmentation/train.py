import pickle
import os
import torch
import argparse
import neptune
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from resblockunet import ResBlockUnet
from torch.utils.data import DataLoader

from dataset import CTDataset
from dataset_CBCT import CBCTDataset
from dataset_combined import CombinedDataset
from preprocess import ClipAndNormalize, GaussianAdditiveNoise, RandomElastic
from utils.plotting import plot_2d
from mini_model import UNetResBlocks 

import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True

BLADDER = 0
CERVIX = 1
BACKGROUND = 2

def get_loss_func(loss_func="BCE"):
    if loss_func == "BCE":
        return nn.BCEWithLogitsLoss()
    if loss_func == "NLL":
        # Classes: bladder, cervix, other
        # Check class weights
        weights = torch.Tensor([2, 4, 1]).to("cuda")
        if args.topk < 1.0:
            return nn.NLLLoss(weight=weights, reduction="none")
        else:
            return nn.NLLLoss(weight=weights)


class SplitLoss(object):
    def __init__(self, classes=3):
        self.criterions = []
        for i in range(classes):
            w = torch.zeros(classes)
            w[i] = 1
            self.criterions.append(nn.CrossEntropyLoss(weight=w))
    
    def __call__(self, y, y_hat):
        losses = []
        y = y.detach().cpu()
        y_hat = y_hat.detach().cpu()
        for criterion in self.criterions:
            loss = criterion(y_hat, y.argmax(1))
            losses.append(loss)
        return losses


def get_model(args, device):
    logger.info("Loading Model...")
    model = ResBlockUnet(1, 3, (1, 512, 512), use_group_norm=args.use_group_norm)
    if args.load_model:
        model.load_state_dict(torch.load(open(args.model_file, 'rb')))
        logger.info("Model loaded!")
    model.to(device)

    return model


def check_results_folder(args):
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
    seg_arr_bladder = Y[:, BLADDER, :, :, :].squeeze().detach().cpu()
    seg_arr_cervix = Y[:, CERVIX, :, :, :].squeeze().detach().cpu()

    out_arr_bladder = Y_hat.exp()[:, BLADDER, :, :, :].squeeze().detach().cpu()
    out_arr_cervix = Y_hat.exp()[:, CERVIX, :, :, :].squeeze().detach().cpu()

    masked_img_bladder = np.array(
        plot_2d(img_arr, mask=out_arr_bladder, mask_color="r", mask_threshold=0.5))
    masked_img_cervix = np.array(
        plot_2d(img_arr, mask=out_arr_cervix, mask_color="r", mask_threshold=0.5))

    masked_img_bladder = torch.from_numpy(
        np.array(plot_2d(masked_img_bladder, mask=seg_arr_bladder, mask_color="g")))
    masked_img_cervix = torch.from_numpy(
        np.array(plot_2d(masked_img_cervix, mask=seg_arr_cervix, mask_color="g")))

    writer.add_image(
        f"{tag}/bladder", Y_hat.exp()[:, BLADDER, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/cervix", Y_hat.exp()[:, CERVIX, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/bladder_gt", Y[:, BLADDER, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/cervix_gt", Y[:, CERVIX, :, :, :].squeeze(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/mask_bladder", masked_img_bladder, i, dataformats="HWC")
    writer.add_image(
        f"{tag}/mask_cervix", masked_img_cervix, i, dataformats="HWC")


def evaluate(dl_val, writer, model, device, criterion, j, max_iters=None):
    losses = {"all": [], "segs": [], "bladder": [], "cervix": [], "background": []}
    softmax = nn.LogSoftmax(1)
    i = 1
    split_loss = SplitLoss()
    

    model.eval()
    for X, Y in tqdm(dl_val):
        X, Y = X.to(device).float(), Y.to(device).float()

        if args.equal_train:
            if len(Y.argmax(1).unique()) < 2:
                continue
        torch.cuda.empty_cache()
        Y_hat = model(X)
        if args.loss_func == "NLL":
            Y_hat = softmax(Y_hat)
        loss = criterion(Y_hat, Y.argmax(1)).mean()
        split_losses = split_loss(Y, Y_hat)
        
        if args.save_imgs and i % 10 == 0:
            _log_images(X.detach().cpu(), Y.detach().cpu(), Y_hat.detach().cpu(), i, writer, "validation")

        if len(Y.argmax(1).unique()) > 1:
            losses["segs"].append(loss.detach().cpu().item())
        losses["all"].append(loss.detach().cpu().item())
        losses["bladder"].append(split_losses[BLADDER].detach().cpu().item())
        losses["cervix"].append(split_losses[CERVIX].detach().cpu().item())
        losses["background"].append(split_losses[BACKGROUND].detach().cpu().item())
        torch.cuda.empty_cache()
        i += 1
        if max_iters is not None:
            if i > max_iters:
                break

    loss = np.mean(losses["all"])
    loss_segs = np.mean(losses["segs"])
    return loss, loss_segs, (np.mean(losses["bladder"]), np.mean(losses["cervix"]), np.mean(losses["background"]))


def train(model, dl, dl_val, optimizer, criterion, args, writer, device, j, true_i, losses):
    softmax = nn.LogSoftmax(1)
    tmp_losses = []
    
    model.train()
    for i, (X, Y) in enumerate(dl):
        X, Y = X.to(device).float(), Y.to(device).float()

        if args.mc_train:
            if len(Y.argmax(1).unique()) < 2:
                continue
        if args.equal_train:
            if len(Y.argmax(1).unique()) < 2 and np.random.rand(1) > 0.25:
                continue

        Y_hat = model(X)
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
            eval_loss = evaluate(dl_val, writer, model, device, criterion, j, max_iters=500)
            writer.add_scalar("loss/validation", eval_loss[1], true_i)
            writer.add_scalar("loss/validation/bladder", eval_loss[2][0], true_i)
            writer.add_scalar("loss/validation/cervix", eval_loss[2][1], true_i)
            writer.add_scalar("loss/validation/background", eval_loss[2][2], true_i)
            neptune.send_metric("loss/validation", true_i, eval_loss[1])
            neptune.send_metric("loss/validation/bladder", true_i, eval_loss[2][0])
            neptune.send_metric("loss/validation/cervix", true_i, eval_loss[2][1])
            neptune.send_metric("loss/validation/background", true_i, eval_loss[2][2])
            logger.info("Iteration: {}/{} Validation Loss: {}".format(true_i,
                                                            args.max_iters, eval_loss[1]))
            if eval_loss[1] < losses["best"]:
                losses["best"] = eval_loss[1]
                model_fn = args.model_dir / "best_model_{}.pt".format(args.loss_func)
                torch.save(model.state_dict(), model_fn)
                logger.info("Best model saved: best_model_{}.pt".format(args.loss_func))
            losses["validation"].append(eval_loss[1])

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
                break

        true_i += 1

    return losses, true_i


def main(args):
    check_results_folder(args)
    device = "cuda"  # Run on GPU

    # Load datasets
    files_CBCT_train = pickle.load(
        open("files_CBCT_train.p", 'rb'))  # training
    files_CBCT_val = pickle.load(
        open("files_CBCT_validation.p", 'rb'))  # validation

    # Load datasets
    files_CT_train = pickle.load(
        open("files_CT_train.p", 'rb'))  # training
    files_CT_val = pickle.load(
        open("files_CT_validation.p", 'rb'))  # validation
    print(f"file lengths: {len(files_CBCT_train)}, {len(files_CT_train)}")

    transform_CBCT= transforms.Compose(
        [GaussianAdditiveNoise(0, 10), RandomElastic((21,512,512)), ClipAndNormalize(750, 1250)])#, RandomElastic((21,512,512))])
    transform_CT= transforms.Compose(
        [GaussianAdditiveNoise(0, 10), RandomElastic((21,512,512)), ClipAndNormalize(-100, 300)])#, RandomElastic((21,512,512))])
    ds_CT = CTDataset(files_CT_train, transform=transform_CT)
    ds_CBCT = CBCTDataset(files_CBCT_train, transform=transform_CBCT, clipped=False)
    ds_combined = CombinedDataset(ds_CT, ds_CBCT)
    dl = DataLoader(ds_combined, batch_size=1, shuffle=args.shuffle, num_workers=6)

    # ds_CT = CTDataset(files_CT_val, transform=transform_CT)
    ds_CBCT = CBCTDataset(files_CBCT_val, transform=transform_CBCT, clipped=False)
    # ds_combined = CombinedDataset(ds_CT, ds_CBCT)
    dl_val = DataLoader(ds_CBCT, batch_size=1, shuffle=args.shuffle, num_workers=6)

    writer = SummaryWriter()

    model = get_model(args, device)
    logger.info("Model Loaded")
    optimizer = Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

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

    with (args.results_folder / "config.txt").open("w") as f:
        f.write(str(args))

    logger.info("Start Training...")
    true_i = 0
    for j in range(args.max_epochs):
        losses, true_i = train(model, dl, dl_val, optimizer,
                               criterion, args, writer, device, j, true_i, losses)

        if true_i > args.max_iters:
            break

        if j > 14:
            args.mc_train = False
        # scheduler.step()
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
                        default="NLL", required=False, type=str)
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
    main(args)

    neptune.append_tag("finished")
    neptune.stop()