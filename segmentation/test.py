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
    
    # Dataset for ONE CT image
    image_shapes = pickle.load(open("CBCT_shapes.p", 'rb'))
    cbct = patients[1] + "\\X01"
    print(cbct)
    image_shapes_0 = {cbct: image_shapes[cbct]}
        
    ds = CervixDataset(patients[1:2], args.root_dir, image_shapes_0, conebeams=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    writer = SummaryWriter()

    '''
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    '''

    print("Loading Model...")
    model = ResBlockUnet(1, 3, (1, 512, 512))
    if args.load_model:
        model.load_state_dict(torch.load(open(args.model_file, 'rb')))
        print("Model loaded!")
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = nn.BCELoss()
    # criterion = DiceLoss()
    softmax = nn.Sigmoid()

    losses = []
    best_loss = float("inf")

    print("Start Training...")
    image_losses = []
    img_losses = []

    final_segmentation = []
    j=0
    for i, (X, Y) in enumerate(dl):
        model.eval()
        X, Y = X.to(device).float(), Y.to(device).float()
        X = X - X.mean()

        Y_hat = model(X)
        assert Y_hat.shape == Y.shape, "output and classification must be same shape, {}, {}".format(Y_hat.shape, Y.shape)
        Y_hat = softmax(Y_hat)
        loss = criterion(Y_hat, Y)
        losses.append(loss.item())
        img_losses.append(loss.item())
        writer.add_scalar("loss/train", loss.item(), j * len(dl) + i)

        writer.add_image("images_true/0", Y[:,0,:,:,:].squeeze(), i, dataformats="HW")
        writer.add_image("images_true/1", Y[:,1,:,:,:].squeeze(), i, dataformats="HW")
        writer.add_image("images_true/2", Y[:,2,:,:,:].squeeze(), i, dataformats="HW")


        writer.add_image("images/0", Y_hat[:,0,:,:,:].squeeze(), i, dataformats="HW")
        writer.add_image("images/1", Y_hat[:,1,:,:,:].squeeze(), i, dataformats="HW")
        writer.add_image("images/2", Y_hat[:,2,:,:,:].squeeze(), i, dataformats="HW")

        torch.cuda.empty_cache()

        if (j * len(dl) + i) % args.print_every == 0:
            print("Iteration: {}/{} Loss: {}".format(j * len(dl) + i, len(dl) * args.max_iters, loss.item()))


        if (j * len(dl) + i) % args.save_every == 0:
            # model.save_state_dict('model.pt')
            save = 1
            # Save Model

    avg_loss = sum(img_losses) / len(img_losses)
    image_losses.append(avg_loss)
    writer.add_scalar("loss/image", avg_loss, j)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pt")

    print("Iteration: {}/{} Average Loss: {}".format(j * len(dl) + i, len(dl) * args.max_iters, avg_loss))
     
    print("End training, save final model...")
    torch.save(model.state_dict(), "final_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data", default="/data/cervix/patients", required=False)
    parser.add_argument("-model_file", help="Get the file containing the model weights", default="final_model.pt", required=False)
    parser.add_argument("-load_model", help="Get the model weights", default="False", required=False, action="store_true")

    parser.add_argument("-lr", help="Learning Rate", default=0.0001, required=False, type=float)
    parser.add_argument("-mc_train", help="Only train using images with at least 2 classes", default=False, required=False, action="store_true")

    parser.add_argument("-max_iters", help="Maximum number of iterations", default=1, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations", default=10, required=False, type=int)
    parser.add_argument("-save_every", help="Save model every X iterations", default=10, required=False, type=int)
    
    args = parser.parse_args()

    print(args)

    train(args)

