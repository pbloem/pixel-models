import os, tqdm, random, pickle

import torch
import torchvision

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU
from torch.optim import Adam

import nltk

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

import util

from tensorboardX import SummaryWriter

from layers import PlainMaskedConv2d

# CIFAR dimensions
C, W, H = 3, 32, 32


"""
TODO:
 - Gated activation.
 - Eliminate blind spot.

"""

def go(arg):

    ## Load the data

    trainset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=True,
                                            download=True, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=False,
                                           download=True, transform=ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                             shuffle=False, num_workers=2)

    ## Set up the model
    fm = 2
    model = Sequential(
        PlainMaskedConv2d(False, 3,  fm, 3, 1, 1, bias=False), BatchNorm2d(fm), ReLU(True),
        # PlainMaskedConv2d(True, fm, fm, 3, 1, 1, bias=False), BatchNorm2d(fm), ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        # MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        Conv2d(fm, 256*C, 1),
        util.Reshape((256, C, W, H))
    )

    print('Constructed network', model)

    # A sample of 144 square images with 3 channels, of the chosen resolution
    # (144 so we can arrange them in a 12 by 12 grid)
    sample = torch.Tensor(144, 3, W, H)

    optimizer = Adam(model.parameters(), lr=arg.lr)

    if torch.cuda.is_available():
        model.cuda()
        sample = sample.cuda()

    for epoch in range(arg.epochs):

        # Train
        err_tr = []
        model.train(True)

        for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

            # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()

            target = (input.data * 255).long()

            input, target = Variable(input), Variable(target)

            # Forward pass
            result = model(input)

            loss = cross_entropy(result, target)

            err_tr.append(loss.data.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        # - we evaluate on the test set, since this is only a simpe reproduction experiment
        #   make sure to split off a validation set if you want to tune hyperparameters for something important

        err_te = []
        model.train(False)

        for i, (input, _) in enumerate(tqdm.tqdm(testloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

            if torch.cuda.is_available():
                input = input.cuda()

            target = (input.data * 255).long()
            input, target = Variable(input), Variable(target)

            result = model(input)
            loss = cross_entropy(result, target)

            err_te.append(loss.data.item())

        # Sample
        sample.fill_(0)
        model.train(False)
        for i in tqdm.trange(W):
            for j in range(H):
                for c in range(C):
                    result = model(Variable(sample, volatile=True))
                    probs = softmax(result[:, :, c, i, j]).data

                    pixel_sample = torch.multinomial(probs, 1).float() / 255.
                    sample[:, c, i, j] = pixel_sample.squeeze()

        utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)

        print('epoch={:02}; training loss: {:.3f}; test loss: {:.3f}'.format(
            epoch, sum(err_tr)/len(err_tr), sum(err_te)/len(err_te)))


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-L", "--latent-size",
                        dest="latent_size",
                        help="Size of the latent representations.",
                        default=32, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the embeddings.",
                        default=300, type=int)

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit on the number of instances seen per epoch (for debugging).",
                        default=None, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate.",
                        default=0.001, type=float)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/score', type=str)

    parser.add_argument("-C", "--cache-directory",
                        dest="cache_dir",
                        help="Dir for cache files (delete the dir to reconstruct)",
                        default='./cache', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)