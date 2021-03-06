import pickle
import torch
import argparse
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from utils.image_readers import read_image
from utils.image_writers import write_image
from utils.plotting import plot_2d
from models.XY_models import Ynetwork
from torch.utils.data import DataLoader
from preprocess import ClipAndNormalize
from datasets.dataset_duo import DuoDataset
from datetime import datetime
from utils.nikolov_metrics import *
import skimage
from skimage import measure

import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


torch.backends.cudnn.benchmark = True


def _log_images(X, Y, Y_hat, i, writer, tag="train"):
    middle_slice = X.shape[2] // 2
    img_arr = X[0, 0, middle_slice, :, :].detach().cpu().numpy()
    seg_arr_bladder = Y[:, 0, :, :, :].squeeze().detach().cpu().numpy()
    seg_arr_cervix = Y[:, 1, :, :, :].squeeze().detach().cpu().numpy()

    out_arr_bladder = Y_hat.exp()[:, 0, :, :, :].squeeze().detach().cpu().numpy()
    out_arr_cervix = Y_hat.exp()[:, 1, :, :, :].squeeze().detach().cpu().numpy()

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
        f"{tag}/bladder_gt", Y[:, 0, :, :, :].squeeze().float(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/cervix_gt", Y[:, 1, :, :, :].squeeze().float(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/other_gt", Y[:, 2, :, :, :].squeeze().float(), i, dataformats="HW")
    writer.add_image(
        f"{tag}/mask_bladder", masked_img_bladder, i, dataformats="HWC")
    writer.add_image(
        f"{tag}/mask_cervix", masked_img_cervix, i, dataformats="HWC")


def iou(ground_truth, segmentation, threshold=0.5, eps=1e-5):
    segmentation = (segmentation > threshold)
    ground_truth = (ground_truth > threshold)
    intersect = (ground_truth & segmentation).sum().item()
    union = (ground_truth | segmentation).sum().item()
    return (intersect + eps) / (union + eps)


def dice(ground_truth, segmentation, threshold=0.5, eps=1e-6):
    segmentation = (segmentation > threshold)
    ground_truth = (ground_truth > threshold)
    intersect = (ground_truth & segmentation).sum().item()
    union = (ground_truth | segmentation).sum().item()
    return (2*intersect + eps) / (intersect + union + eps)


def get_loss_func(loss_func="BCE"):
    if loss_func == "BCE":
        return nn.BCEWithLogitsLoss()
    if loss_func == "NLL":
        weights = torch.Tensor([1, 1, 0.1]).to("cuda")
        return nn.NLLLoss(weight=weights)


def calculate_metrics(mask_gt, mask_pred, spacing_mm, percent, tolerances):
    metrics = {}
    surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm)
    metrics["avg_surface_distance"] = compute_average_surface_distance(surface_distances)
    metrics["robust_hausdorff"] = compute_robust_hausdorff(surface_distances, percent)
    for tolerance_mm in tolerances:
        metrics[f"surface_overlap_tolerance_{tolerance_mm}"] = compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm)
        metrics[f"surface_dice_tolerance_{tolerance_mm}"] = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)
    metrics["dice"] = compute_dice_coefficient(mask_gt, mask_pred)
    return metrics


def test(args, dl, writer, model):
    device = "cuda"  # Run on GPU

    criterion = get_loss_func("NLL")
    softmax = nn.LogSoftmax(1)

    logger.info("Start Testing...")
    tmp_losses = []
    metrics = {"bladder": {}, "cervix": {}}

    segmentations = {0: [], 1: [], "y_bladder": [], "y_cervix": [], "cbct": []}

    img_i = 0
    reset_shape = True

    all_zeros = 0
    seg_slices = 0

    model.eval()
    for i, (metadata, patient, X_cbct, X_ct, Y_cbct, Y_ct) in enumerate(dl):
        X_cbct, X_ct, Y_cbct = X_cbct.to(device).float(), X_ct.to(device).float(), Y_cbct.to(device).float()
        
        if reset_shape:
            img_shape = metadata["depth"]
            logger.debug(f"New Shape: {img_shape} for patient: {patient[0]}")
            reset_shape = False
        torch.cuda.empty_cache()
        Y_hat = model(X_cbct, X_ct)
        assert Y_hat.shape == Y_cbct.shape, "output and classification must be same shape, {}, {}".format(
            Y_hat.shape, Y_cbct.shape)

        if args.save_3d:
            middle_slice = X_cbct.shape[2] // 2
            segmentations["y_bladder"].append(
                Y_cbct[:, 0, :, :, :].squeeze().detach().cpu())
            segmentations["y_cervix"].append(
                Y_cbct[:, 1, :, :, :].squeeze().detach().cpu())
            segmentations["cbct"].append(
                X_cbct[0, 0, middle_slice, :, :].squeeze().detach().cpu())

        Y_hat = softmax(Y_hat)
        loss = criterion(Y_hat, Y_cbct.argmax(1))
        tmp_losses.append(loss.detach().cpu().item())

        segmentations[0].append(
            Y_hat.exp()[:, 0, :, :, :].squeeze().detach().cpu() > 0.5)
        segmentations[1].append(
            Y_hat.exp()[:, 1, :, :, :].squeeze().detach().cpu() > 0.5)

        img_i += 1

        if img_i >= img_shape and args.save_3d:
            img_i = 0
            patient = patient[0]
            logger.info(f"Saving image {patient}, image size: {img_shape}")

            if not isinstance(metadata["orig_spacing"], tuple): metadata["orig_spacing"] = tuple(*metadata["orig_spacing"].tolist())
            if not isinstance(metadata["spacing"], tuple): metadata["spacing"] = tuple([j.item() for j in metadata["spacing"]])
            if not isinstance(metadata["orig_origin"], tuple): metadata["orig_origin"] = tuple([j.item() for j in metadata["orig_origin"]])
            if not isinstance(metadata["origin"], tuple): metadata["origin"] = tuple([j.item() for j in metadata["origin"]])
            if not isinstance(metadata["orig_direction"], tuple): metadata["orig_direction"] = tuple([j.item() for j in metadata["orig_direction"]])
            if not isinstance(metadata["direction"], tuple): metadata["direction"] = tuple([j.item() for j in metadata["direction"]])
            if  isinstance(metadata["filename"], list): metadata["filename"] = metadata["filename"][0]

            cbct = torch.stack(segmentations["cbct"]).detach().cpu().numpy()
            y_bladder = torch.stack(segmentations["y_bladder"]).detach().cpu().numpy()
            y_cervix = torch.stack(segmentations["y_cervix"]).detach().cpu().numpy()
            seg_bladder = torch.stack(segmentations[0]).detach().cpu().numpy()
            seg_cervix = torch.stack(segmentations[1]).detach().cpu().numpy()

            if args.post_process:
                labels_mask = measure.label(seg_bladder)
                regions = measure.regionprops(labels_mask)
                regions.sort(key=lambda x: x.area, reverse=True)
                if len(regions) > 1:
                    for rg in regions[1:]:
                        labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
                labels_mask[labels_mask!=0] = 1
                seg_bladder = labels_mask

                labels_mask = measure.label(seg_cervix)
                regions = measure.regionprops(labels_mask)
                regions.sort(key=lambda x: x.area, reverse=True)
                if len(regions) > 1:
                    for rg in regions[1:]:
                        labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
                labels_mask[labels_mask!=0] = 1
                seg_cervix = labels_mask

            ref_fn = metadata["filename"]
            write_image(seg_bladder.astype(np.uint8), "{}/{}_seg_bladder.nrrd".format(
                args.test_folder, patient.replace("/", "_")), metadata=metadata, ref_fn=True)
            write_image(seg_cervix.astype(np.uint8), "{}/{}_seg_cervix_uterus.nrrd".format(
                args.test_folder, patient.replace("/", "_")), metadata=metadata, ref_fn=True)
            metrics_bladder = calculate_metrics(y_bladder.astype(bool), seg_bladder.astype(bool), metadata["spacing"], 25.0, [0.5, 1.0, 1.5, 3.0])
            metrics_cervix = calculate_metrics(y_cervix.astype(bool), seg_cervix.astype(bool), metadata["spacing"], 25.0, [0.5, 1.0, 1.5, 3.0])

            metrics["bladder"][patient] = metrics_bladder
            metrics["cervix"][patient] = metrics_cervix

            for m, v in metrics_bladder.items():
                logger.info(f"{m} bladder: {v}")
            for m, v in metrics_cervix.items():
                logger.info(f"{m} cervix: {v}")
            
            segmentations = {0: [], 1: [], "y_bladder": [], "y_cervix": [], "cbct": []}
            reset_shape = True

        torch.cuda.empty_cache()

        _log_images(X_cbct, Y_cbct, Y_hat, i, writer, tag=patient[0])

        if i % args.print_every == 0:
            logger.info("Iteration: {}/{} Loss: {}".format(i+1,
                                                           len(dl), sum(tmp_losses)/len(tmp_losses)))
            tmp_losses = []
    
    pickle.dump(metrics, open("{}/metrics.p".format(args.test_folder), 'wb'))

    writer.flush()
    logger.info("End testing")


def main(args):
    device = "cuda"
    image_shapes = pickle.load(open("files_sCT_pCT_test.p", 'rb'))

    transform_CBCT= transforms.Compose(
        [ClipAndNormalize(800, 1250)])
    ds = DuoDataset(image_shapes, transform=transform_CBCT, n_slices=21, return_meta=True)
    if args.save_3d:
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=12)
    else:
        dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=12)

    writer = SummaryWriter()

    logger.info("Loading Model...")
    model = Ynetwork(1, 3, (1, 512, 512))
    model.load_state_dict(torch.load(open(args.model_file, 'rb')))
    logger.info("Model loaded!")
    model.to(device)
    model.eval()

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

    test(args, dl, writer, model)


def parse_args():

    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-model_file", help="Get the file containing the model weights",
                        default="final_model.pt", required=False)

    parser.add_argument("-print_every", help="Print every X iterations",
                        default=10, required=False, type=int)
    parser.add_argument("-save_3d", help="Save 3D segmentations",
                        default=False, required=False, action="store_true")
    parser.add_argument("-post_process", help="Post-process segmetnation",
                        default=False, required=False, action="store_true")
    parser.add_argument("-test_folder", help="Where to save testing results",
                        default="testing", required=False)
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
