import os, tqdm, random, pickle

import torch
import torchvision

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize, Grayscale
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
import torch.nn.functional as F
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU
from torch.optim import Adam

# import nltk

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

import util, models

from tensorboardX import SummaryWriter

from layers import PlainMaskedConv2d, MaskedConv2d

SEEDFRAC = 2


def draw_sample(seeds, model, seedsize=(0, 0), batch_size=32):

    b, c, h, w = seeds.size()

    sample = seeds.clone()
    if torch.cuda.is_available():
        sample = sample.cuda()
    sample = Variable(sample)

    for i in tqdm.trange(h):
        for j in range(w):
            if i < seedsize[0] and j < seedsize[1]:
                continue

            for channel in range(c):

                result = util.batched(sample, model, batch_size)

                probs = softmax(result[:, :, channel, i, j])

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

    elif arg.task == 'cifar-gs':
        transform = Compose([Grayscale(), ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 32, 32

    elif arg.task == 'imagenet64':
        transform = Compose([ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 64, 64

    else:
        raise Exception('Task {} not recognized.'.format(arg.task))

    ## Set up the model
    fm = arg.channels
    krn = arg.kernel_size
    pad = krn // 2

    if arg.model == 'simple':

        modules = []
        for i in range(arg.num_layers):
            modules.append(
                PlainMaskedConv2d(i > 0,  fm if i > 0 else C, fm, krn, 1, pad, bias=False))

            modules.append(ReLU(True))

        modules.extend([
            Conv2d(fm, 256 * C, 1),
            util.Reshape((256, C, W, H))
        ])

        model = Sequential(*modules)

    elif arg.model == 'gated-old':

        modules = [
            Conv2d(C, fm, 1, groups=C),  # the groups allow us to block out certain colors in the first layer
            util.Lambda(lambda x: (x, x))
        ]

        for i in range(arg.num_layers):
            modules.append(MaskedConv2d(fm, colors=C, self_connection=i > 0,
                                         res_connection=(not arg.no_res) if i > 0 else False,
                                         gates=not arg.no_gates,
                                         hv_connection=not arg.no_hv,
                                         k=krn, padding=pad))

        modules.extend([
            util.Lambda(lambda xs: xs[1]),
            Conv2d(fm, 256*C, 1, groups=C),
            util.Reshape((C, 256, W, H)),
            util.Lambda(lambda x: x.transpose(1, 2)) # index for batched tensor
        ])

        model = Sequential(*modules)

    elif arg.model == 'gated':

        model = models.Gated((C, H, W), arg.channels,
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
    testbatch = util.readn(testloader, n=12)
    testbatch = testbatch.unsqueeze(1).expand(12, 6, C, H, W).contiguous().view(72, 1, C, H, W).squeeze(1)
    sample_init_seeds[:, :, :sh, :] = testbatch[:, :, :sh, :]

    optimizer = Adam(model.parameters(), lr=arg.lr)

    if torch.cuda.is_available():
        model.cuda()

    instances_seen = 0
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

            loss = loss * util.LOG2E  # Convert from nats to bits

            instances_seen += input.size(0)
            tbw.add_scalar('pixel-models/training-loss', float(loss.data.item()), instances_seen)
            err_tr.append(float(loss.data.item()))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del loss, result

        # Evaluate
        # - we evaluate on the test set, since this is only a simple reproduction experiment
        #   make sure to split off a validation set if you want to tune hyperparameters for something important

        if epoch % arg.eval_every == 0 and epoch != 0:
            with torch.no_grad():

                err_test = 0.0
                err_total = 0

                model.train(False)

                for i, (input, _) in enumerate(tqdm.tqdm(testloader)):
                    if arg.limit is not None and i * arg.batch_size > arg.limit:
                        break

                    if torch.cuda.is_available():
                        input = input.cuda()

                    target = (input.data * 255).long()
                    input, target = Variable(input), Variable(target)

                    result = model(input)
                    loss = cross_entropy(result, target, reduction='none')

                    loss = loss * util.LOG2E # Convert from nats to bits

                    err_test += float(loss.data.sum())
                    err_total += util.prod(input.size())

                del loss, result

                testloss = err_test/err_total

                tbw.add_scalar('pixel-models/test-loss', testloss, epoch)
                print('epoch={:02}; training loss: {:.3f}; test loss: {:.3f}'.format(
                    epoch, sum(err_tr)/len(err_tr), testloss))

                # Compute loss pixel by pixel, color by color, to make sure we're not leaking

                if arg.pbp:

                    sum_bits = 0
                    total = 0
                    for i, (input, _) in enumerate(tqdm.tqdm(testloader)):

                        mask = torch.zeros(*input.size())

                        if torch.cuda.is_available():
                            input, mask = input.cuda(), mask.cuda()

                        target = (input.data * 255).long()

                        input = Variable(input)

                        for h in range(H):
                            for w in range(W):
                                for c in range(C):

                                    result = model(input * mask)
                                    result = F.log_softmax(result, dim=1)

                                    for b in range(input.size(0)):
                                        t = target[b, c, h, w]
                                        sum_bits += - float(result[b, t, c, h, w].sum())
                                        total += 1

                                    mask[:, c, h, w] += 1

                    print('epoch={:02}; pixel-by-pixel test loss: {:.3f}'.format(
                        epoch,sum_bits/total))

                model.train(False)
                sample_zeros = draw_sample(sample_init_zeros, model, seedsize=(0, 0), batch_size=arg.batch_size)
                sample_seeds = draw_sample(sample_init_seeds, model, seedsize=(sh, W), batch_size=arg.batch_size)
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
                        help="Type of model to use: [simple, gated].",
                        default='simple', type=str)

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

    parser.add_argument("--pixel-by-pixel",
                        dest="pbp",
                        help="Compute a pixel-by-pixel loss on the test set every epoch. Slow, but more certain to be fair than the plain test loss.",
                        action='store_true')

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=15, type=int)

    parser.add_argument("--evaluate-every",
                        dest="eval_every",
                        help="Run an exaluation/sample every n epochs.",
                        default=1, type=int)

    parser.add_argument("-k", "--kernel_size",
                        dest="kernel_size",
                        help="Size of convolution kernel",
                        default=7, type=int)

    parser.add_argument("-x", "--num-layers",
                        dest="num_layers",
                        help="Number of extra convolution layers (after the first one)",
                        default=7, type=int)

    parser.add_argument("-c", "--channels",
                        dest="channels",
                        help="Number of channels (aka feature maps) for the intermediate representations. Should be divisible by the number of colors.",
                        default=63, type=int)

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