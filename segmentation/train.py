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


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

def train(args):
    patients = os.listdir(args.root_dir)

    device = "cuda" # Run on GPU

    # Dataset for CBCTs
    # image_shapes = pickle.load(open("CBCT_shapes.p", 'rb'))
    # patient = list(image_shapes.keys())[0]
    # image_shapes_0 = {patient: image_shapes[patient]}
       
    # ds = CervixDataset(args.root_dir, image_shapes_0, conebeams=True)
    
    # Dataset for ONE CT image
    image_shapes = pickle.load(open("CT_shapes.p", 'rb'))
    # cbct = patients[0] + "\\X01"
    # image_shapes_0 = {cbct: image_shapes[cbct]}
        
    ds = CervixDataset(args.root_dir, image_shapes, conebeams=False)
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

    criterion = nn.BCEWithLogitsLoss()
    softmax = nn.Sigmoid()

    losses = []
    best_loss = float("inf")

    print("Start Training...")
    image_losses = []
    for j in range(args.max_iters):
        img_losses = []
        for i, (X, Y) in enumerate(dl):
            model.train()
            X, Y = X.to(device).float(), Y.to(device).float()
            X = X - X.mean()

            if args.mc_train:
                if len(Y.argmax(1).unique()) < 2:
                    continue

            Y_hat = model(X)
            assert Y_hat.shape == Y.shape, "output and classification must be same shape"
            loss = criterion(Y_hat, Y)
            
            Y_hat = softmax(Y_hat)

            losses.append(loss.item())
            img_losses.append(loss.item())
            writer.add_scalar("loss/slice", loss.item(), j * len(dl) + i)


            # if j%50 == 0:
            #     writer.add_image("images_true/0", Y[:,0,:,:,:].squeeze(), j * len(dl) + i, dataformats="HW")
            #     writer.add_image("images_true/1", Y[:,1,:,:,:].squeeze(), j * len(dl) + i, dataformats="HW")
            #     writer.add_image("images_true/2", Y[:,2,:,:,:].squeeze(), j * len(dl) + i, dataformats="HW")


            #     writer.add_image("images/0", Y_hat[:,0,:,:,:].squeeze(), j * len(dl) + i, dataformats="HW")
            #     writer.add_image("images/1", Y_hat[:,1,:,:,:].squeeze(), j * len(dl) + i, dataformats="HW")
            #     writer.add_image("images/2", Y_hat[:,2,:,:,:].squeeze(), j * len(dl) + i, dataformats="HW")

            torch.cuda.empty_cache()

            if (j * len(dl) + i) % args.print_every == 0:
                print("Iteration: {}/{} Loss: {}".format(j * len(dl) + i, len(dl) * args.max_iters, loss.item()))

            # Train model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = sum(img_losses) / len(img_losses)
        image_losses.append(avg_loss)
        writer.add_scalar("loss/image", avg_loss, j)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved")

        print("Iteration: {}/{} Average Loss: {}".format(j * len(dl) + i, len(dl) * args.max_iters, avg_loss))

        if j % args.save_every == 0:
            # Save Model
            torch.save(model.state_dict(), "model_{}.pt".format(j))

        if j == 75:
            args.mc_train = False

     
    print("End training, save final model...")
    torch.save(model.state_dict(), "final_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", help="Get root directory of data", default="/data/cervix/patients", required=False)
    parser.add_argument("-model_file", help="Get the file containing the model weights", default="final_model.pt", required=False)
    parser.add_argument("-load_model", help="Get the model weights", default=False, required=False, action="store_true")

    parser.add_argument("-lr", help="Learning Rate", default=0.0001, required=False, type=float)
    parser.add_argument("-mc_train", help="Only train using images with at least 2 classes", default=False, required=False, action="store_true")

    parser.add_argument("-max_iters", help="Maximum number of iterations", default=10, required=False, type=int)
    parser.add_argument("-print_every", help="Print every X iterations", default=10, required=False, type=int)
    parser.add_argument("-save_every", help="Save model every X iterations", default=10, required=False, type=int)
    
    args = parser.parse_args()

    print(args)

    train(args)

# python train.py -max_iters 100 -save_every 10 -mc_train
# Train on all 10 patient CT images. 