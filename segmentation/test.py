import pickle
import pdb
import os
import torch
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from utils.image_readers import read_image
from utils.image_writers import write_image
from resblockunet import ResBlockUnet
from torch.utils.data import DataLoader
from preprocess import Clip, NormalizeHV
from dataset_extra_CTs import ExtraCervixDataset


def train(args):
    device = "cuda"  # Run on GPU

    # Dataset for ONE CT image
    image_shapes = pickle.load(open("extra_CT_shapes_validation.p", 'rb'))
    cbct = list(image_shapes.keys())[0]
    # cbct = '9700751'
    print(cbct, type(cbct))
    image_shapes_0 = {cbct: image_shapes[cbct]}

    transform = transforms.Compose([Clip(), NormalizeHV()])
    ds = ExtraCervixDataset(args.root_dir, image_shapes_0, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    writer = SummaryWriter()

    _, metadata = read_image("/data/cervix/converted/CervixLoP-1/full/CT.nrrd")
    print(metadata)

    '''
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    '''

    print("Loading Model...")
    model = ResBlockUnet(1, 3, (1, 512, 512))
    model.load_state_dict(torch.load(open(args.model_file, 'rb')))
    print("Model loaded!")
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    softmax = nn.Sigmoid()

    print("Start Testing...")
    img_losses = []

    segmentations = {0: [], 1: []}

    j = 0
    for i, (X, Y) in enumerate(dl):
        torch.cuda.empty_cache()
        # pdb.set_trace()
        X, Y = X.to(device).float(), Y.to(device).float()
        # X = X_2 - 0.7 #normalization

        torch.cuda.empty_cache()
        Y_hat = model(X - 0.7)
        assert Y_hat.shape == Y.shape, "output and classification must be same shape, {}, {}".format(
            Y_hat.shape, Y.shape)

        loss = criterion(Y_hat, Y)
        img_losses.append(loss.item())
        # writer.add_scalar("loss/train", loss.item(), j * len(dl) + i)

        # TODO: overlay images

        writer.add_image(
            "images_true/0", Y[:, 0, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "images_true/1", Y[:, 1, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "images_true/2", Y[:, 2, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "images_true/X", X[:, :, 10:11, :, :].squeeze(), i, dataformats="HW")

        Y_hat = softmax(Y_hat)

        writer.add_image(
            "images/0", Y_hat[:, 0, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "images/1", Y_hat[:, 1, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "images/2", Y_hat[:, 2, :, :, :].squeeze(), i, dataformats="HW")
        writer.add_image(
            "images/X", X[:, :, 10:11, :, :].squeeze(), i, dataformats="HW")

        # TODO: create whole image and segmentation

        # image.append(X[:,:,10:11,:,:].squeeze().data)
        segmentations[0].append(Y_hat[:, 0, :, :, :].squeeze().data > 0.5)
        segmentations[1].append(Y_hat[:, 1, :, :, :].squeeze().data > 0.5)

        torch.cuda.empty_cache()

        if (j * len(dl) + i) % args.print_every == 0:
            print("Iteration: {}/{} Loss: {}".format(j * len(dl) +
                                                     i, len(dl) * args.max_iters, loss.item()))

    seg_bladder = torch.stack(segmentations[0]).cpu().int()
    seg_cervix_uterus = torch.stack(segmentations[1]).cpu().int()

    print(seg_bladder.shape)
    print(seg_cervix_uterus.shape)

    write_image(seg_bladder, "testing/seg_bladder.nrrd", metadata=metadata)
    write_image(seg_cervix_uterus,
                "testing/seg_cervix_uterus.nrrd", metadata=metadata)

    avg_loss = sum(img_losses) / len(img_losses)
    # writer.add_scalar("loss/average", avg_loss, j)

    print("Iteration: {}/{} Average Loss: {}".format(j *
                                                     len(dl) + i, len(dl) * args.max_iters, avg_loss))
    print("End testing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data",
                        default="/data/cervix/patients", required=False)
    parser.add_argument("-model_file", help="Get the file containing the model weights",
                        default="final_model.pt", required=False)

    parser.add_argument("-max_iters", help="Maximum number of iterations",
                        default=1, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations",
                        default=10, required=False, type=int)

    args = parser.parse_args()

    print(args)

    train(args)
