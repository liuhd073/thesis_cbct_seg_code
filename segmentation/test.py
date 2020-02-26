import pickle
import pdb
import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from utils.image_readers import read_image
from utils.image_writers import write_image
from utils.plotting import plot_2d
from resblockunet import ResBlockUnet
from torch.utils.data import DataLoader
from preprocess import Clip, NormalizeHV
from dataset import CTDataset
from mini_model import UNetResBlocks


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


def iou(ground_truth, segmentation, threshold=0.5, eps=1e-5):
    segmentation = (segmentation > threshold)
    ground_truth = (ground_truth > threshold)
    intersect = (ground_truth & segmentation).sum().item()
    union = (ground_truth | segmentation).sum().item()
    return 1 - (intersect + eps) / (union + eps)


def dice(ground_truth, segmentation, threshold=0.5, eps=1e-6):
    segmentation = (segmentation > threshold)
    ground_truth = (ground_truth > threshold)
    intersect = (ground_truth & segmentation).sum().item()
    union = (ground_truth | segmentation).sum().item()
    return 1 - (2*intersect + eps) / (intersect + union + eps)


def get_loss_func(loss_func="BCE"):
    if loss_func == "BCE":
        return nn.BCEWithLogitsLoss()
    if loss_func == "NLL":
        weights = torch.Tensor([1, 1, 0.1]).to("cuda")
        return nn.NLLLoss(weight=weights)


def test(args, dl, writer, model, image_shapes):
    device = "cuda"  # Run on GPU

    criterion = get_loss_func("NLL")
    softmax = nn.LogSoftmax(1)

    print("Start Testing...")
    img_losses = []
    tmp_losses = []

    segmentations = {0: [], 1: [], "y_bladder": [], "y_cervix": []}

    img_i = 0
    temp = image_shapes.pop(0)
    img_shape = temp[1][0]
    patient = temp[0]

    print(os.path.exists(os.path.join(args.root_dir, "converted", patient, "CT.nrrd")))
    print(os.path.join(args.root_dir, "converted", patient, "CT.nrrd"))

    # if os.path.exists(os.path.join(args.root_dir, "converted", patient, "CT.nrrd")):
    #     _, metadata = read_image(os.path.join(args.root_dir, "converted", patient, "CT.nrrd"))
    # else: 
    #     _, metadata = read_image(os.path.join(args.root_dir, "patients", patient, "CT1.nii"))

    print(patient.replace("/", "_"))

    j = 0
    for i, (X, Y) in enumerate(dl):
        if img_i >= img_shape and args.save_3d:
            img_i = 0
            print("NEW IMAGE")

            y_bladder = torch.stack(segmentations["y_bladder"])
            y_cervix = torch.stack(segmentations["y_cervix"])
            seg_bladder = torch.stack(segmentations[0]).int()
            seg_cervix_uterus = torch.stack(segmentations[1]).int()

            write_image(seg_bladder, "testing/{}_seg_bladder.nrrd".format(patient.replace("/", "_")), metadata=metadata)
            write_image(seg_cervix_uterus,
                    "testing/{}_seg_cervix_uterus.nrrd".format(patient.replace("/", "_")), metadata=metadata)

            print(y_bladder.shape, type(y_bladder), seg_bladder.shape, type(seg_bladder))
            pickle.dump((y_bladder, seg_bladder), open("temp.p", 'wb'))

            print("IoU bladder:", iou(y_bladder, seg_bladder, threshold=0.8))
            print("IoU cervix/uterus:", iou(y_cervix, seg_cervix_uterus, threshold=0.8))
            print("Dice bladder:", dice(y_bladder, seg_bladder, threshold=0.8))
            print("Dice cervix/uterus:", dice(y_cervix, seg_cervix_uterus, threshold=0.8))

            segmentations = {0: [], 1: [], "y_bladder": [], "y_cervix": []}
            temp = image_shapes.pop(0)
            img_shape = temp[1][0]
            patient = temp[0]

            if os.path.exists(os.path.join(args.root_dir, "converted", patient, "CT.nrrd")):
                _, metadata = read_image(os.path.join(args.root_dir, "converted", patient, "CT.nrrd"))
            else: 
                _, metadata = read_image(os.path.join(args.root_dir, "patients", patient, "CT1.nii"))


        torch.cuda.empty_cache()
        # pdb.set_trace()
        X, Y = X.to(device).float(), Y.to(device).float()
        # X = X_2 - 0.7 #normalization

        torch.cuda.empty_cache()
        Y_hat = model(X - 0.1)
        assert Y_hat.shape == Y.shape, "output and classification must be same shape, {}, {}".format(
            Y_hat.shape, Y.shape)

        if args.save_3d:
            segmentations["y_bladder"].append(Y[:, 0, :, :, :].squeeze().detach().cpu())
            segmentations["y_cervix"].append(Y[:, 1, :, :, :].squeeze().detach().cpu())

        Y_hat = softmax(Y_hat)
        loss = criterion(Y_hat, Y.argmax(1))
        tmp_losses.append(loss.detach().cpu().item())
        # writer.add_scalar("loss/train", loss.item(), j * len(dl) + i)

        segmentations[0].append(Y_hat.exp()[:, 0, :, :, :].squeeze().detach().cpu() > 0.8)
        segmentations[1].append(Y_hat.exp()[:, 1, :, :, :].squeeze().detach().cpu() > 0.8)
        
        torch.cuda.empty_cache()

        _log_images(X, Y, Y_hat, i, writer, tag="test")

        if img_i % args.print_every == 0:
            print("Epoch: {}/{} Iteration: {}/{} Loss: {}".format(j,
                                                                  args.max_epochs, i, len(dl), sum(tmp_losses)/len(tmp_losses)))
            tmp_losses = []

        img_i += 1

    writer.flush()
    print("End testing")


def main(args):
    device = "cuda"
    # Dataset for ONE CT image
    image_shapes = pickle.load(open("files_test.p", 'rb'))
    
    transform = transforms.Compose([Clip(), NormalizeHV()])
    ds = CTDataset(image_shapes, transform=transform)
    if args.save_3d:
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=12)
        # _, metadata = read_image("/data/cervix/converted/CervixLoP-1/full/CT.nrrd")
        # print(metadata)
    else: 
        dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=12)

    writer = SummaryWriter()
    
    print("Loading Model...")
    # model = ResBlockUnet(1, 3, (1, 512, 512))
    model = UNetResBlocks()
    model.load_state_dict(torch.load(open(args.model_file, 'rb')))
    print("Model loaded!")
    model.to(device)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(pytorch_total_params)

    test(args, dl, writer, model, image_shapes)


def parse_args():

    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix", required=False)
    parser.add_argument("-model_file", help="Get the file containing the model weights",
                        default="final_model.pt", required=False)

    parser.add_argument("-max_epochs", help="Maximum number of epochs",
                        default=1, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations",
                        default=10, required=False, type=int)
    parser.add_argument("-save_3d", help="Save 3D segmentations",
                        default=False, required=False, action="store_true")
    

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
