import os, tqdm, random, pickle

import torch
import torchvision

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU
from torch import nn
from torch.optim import Adam

# import nltk

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

import util, models

from tensorboardX import SummaryWriter

from layers import PlainMaskedConv2d, MaskedConv2d, CMaskedConv2d

SEEDFRAC = 2

def draw_sample(seeds, classes, model, seedsize=(0, 0)):

    b, c, h, w = seeds.size()

    sample = seeds.clone()

    if torch.cuda.is_available():
        sample, classes = sample.cuda(), classes.cuda()

    sample, classes = Variable(sample), Variable(classes)

    for i in tqdm.trange(h):
        for j in range(w):
            if i < seedsize[0] and j < seedsize[1]:
                continue

            for channel in range(c):
                result = model(sample, classes)
                probs = softmax(result[:, :, channel, i, j]).data

                pixel_sample = torch.multinomial(probs, 1).float() / 255.
                sample[:, channel, i, j] = pixel_sample.squeeze()

    return sample

def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    ## Load the data
    if arg.task == 'mnist':
        trainset = torchvision.datasets.MNIST(root=arg.data_dir, train=True,
                                                download=True, transform=ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=arg.data_dir, train=False,
                                               download=True, transform=ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 28, 28
        CLS = 10

    elif arg.task == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=True,
                                                download=True, transform=ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=False,
                                               download=True, transform=ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 32, 32
        CLS = 10
    else:
        raise Exception('Task {} not recognized.'.format(arg.task))

    ## Set up the model

    if arg.model == 'gated':

        model = models.CGated((C, H, W), (CLS,), arg.channels,
                             num_layers=arg.num_layers, k=arg.kernel_size, padding=arg.kernel_size//2)

    else:
        raise Exception('model "{}" not recognized'.format(arg.model))

    print('Constructed network', model)

    # A sample of 144 square images with 3 channels, of the chosen resolution
    # (144 so we can arrange them in a 12 by 12 grid)
    sample_init_zeros = torch.zeros(72, C, H, W)
    sample_init_seeds = torch.zeros(72, C, H, W)

    sh, sw = H//SEEDFRAC, W//SEEDFRAC

    # Init second half of sample with patches from test set, to seed the sampling
    testbatch     = util.readn(testloader, n=12)
    testcls_seeds = util.readn(testloader, n=12, cls=True, maxval=CLS)

    testbatch = testbatch.unsqueeze(1).expand(12, 6, C, H, W).contiguous().view(72, 1, C, H, W).squeeze(1)
    sample_init_seeds[:, :, :sh, :] = testbatch[:, :, :sh, :]
    testcls_seeds = testcls_seeds.unsqueeze(1).expand(12, 6, CLS).contiguous().view(72, 1, CLS).squeeze(1)

    # Get classes for the unseeded part
    testcls_zeros = util.readn(testloader, n=24, cls=True, maxval=CLS)[12:]
    testcls_zeros = testcls_zeros.unsqueeze(1).expand(12, 6, CLS).contiguous().view(72, 1, CLS).squeeze(1)

    if arg.half_precision:
        sample_init_seeds, testcls_seeds = sample_init_seeds.half(), testcls_seeds.half()
        sample_init_zeros, testcls_zeros = sample_init_zeros.half(), testcls_zeros.half()

    optimizer = Adam(model.parameters(), lr=arg.lr)

    if arg.half_precision:
        model.half();

    if torch.cuda.is_available():
        model.cuda()

    instances_seen = 0
    for epoch in range(arg.epochs):

        # Train
        err_tr = []
        model.train(True)

        for i, (input, classes) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

            # Prepare the input
            b, c, w, h = input.size()

            classes = util.one_hot(classes, CLS)

            if arg.half_precision:
                input, classes = input.half(), classes.half()

            if torch.cuda.is_available():
                input, classes = input.cuda(), classes.cuda()

            target = (input.data * 255).long()

            input, classes, target = Variable(input), Variable(classes), Variable(target)

            # Forward pass
            result = model(input, classes)

            loss = cross_entropy(result, target)

            loss = loss * util.LOG2E  # Convert from nats to bits

            instances_seen += input.size(0)
            tbw.add_scalar('pixel-models/training-loss', loss.data.item(), instances_seen)
            err_tr.append(loss.data.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % arg.eval_every == 0 and epoch != 0:
            with torch.no_grad():

                # Evaluate
                # - we evaluate on the test set, since this is only a simpe reproduction experiment
                #   make sure to split off a validation set if you want to tune hyperparameters for something important

                err_te = []
                model.train(False)

                for i, (input, classes) in enumerate(tqdm.tqdm(testloader)):
                    if arg.limit is not None and i * arg.batch_size > arg.limit:
                        break

                    classes = util.one_hot(classes, CLS)

                    if arg.half_precision:
                        input, classes = input.half(), classes.half()

                    if torch.cuda.is_available():
                        input, classes = input.cuda(), classes.cuda()

                    target = (input.data * 255).long()

                    input, classes, target = Variable(input), Variable(classes), Variable(target)

                    result = model(input, classes)
                    loss = cross_entropy(result, target)

                    loss = loss * util.LOG2E  # Convert from nats to bits

                    err_te.append(loss.data.item())

                tbw.add_scalar('pixel-models/test-loss', sum(err_te)/len(err_te), epoch)
                print('epoch={:02}; training loss: {:.3f}; test loss: {:.3f}'.format(
                    epoch, sum(err_tr)/len(err_tr), sum(err_te)/len(err_te)))

                model.train(False)
                sample_zeros = draw_sample(sample_init_zeros, testcls_zeros, model, seedsize=(0, 0))
                sample_seeds = draw_sample(sample_init_seeds, testcls_seeds, model, seedsize=(sh, W))
                sample = torch.cat([sample_zeros, sample_seeds], dim=0)

                utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task: [mnist, cifar10].",
                        default='mnist', type=str)

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Type of model to use: [gated].",
                        default='gated', type=str)

    parser.add_argument("--no-res",
                        dest="no_res",
                        help="Turns off the res connection in the gated layer",
                        action='store_true')

    parser.add_argument("--no-gates",
                        dest="no_gates",
                        help="Turns off the gates in the gated layer",
                        action='store_true')

    parser.add_argument("--no-hv",
                        dest="no_hv",
                        help="Turns off the connection between the horizontal and vertical stack in the gated layer",
                        action='store_true')

    parser.add_argument("--batch-norm",
                        dest="batch_norm",
                        help="Turns on batch normalization after each layer",
                        action='store_true')

    parser.add_argument("--half-precision",
                        dest="half_precision",
                        help="Turns on half-precision (16 bit floats). This doubles the effective memory on compatible GPUs, like the 2080 Ti.",
                        action='store_true')

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("--evaluate-every",
                        dest="eval_every",
                        help="Run an exaluation/sample every n epochs.",
                        default=1, type=int)

    parser.add_argument("-k", "--kernel_size",
                        dest="kernel_size",
                        help="Size of convolution kernel",
                        default=7, type=int)

    parser.add_argument("-x", "--num_layers",
                        dest="num_layers",
                        help="Number of convolution layers",
                        default=7, type=int)

    parser.add_argument("-c", "--channels",
                        dest="channels",
                        help="Number of channels (aka featur maps) for the intermediate representations.",
                        default=60, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

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
                        default='./runs/pixel', type=str)

    parser.add_argument("-C", "--cache-directory",
                        dest="cache_dir",
                        help="Dir for cache files (delete the dir to reconstruct)",
                        default='./cache', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)