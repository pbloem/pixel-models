import os, tqdm, random, pickle

import torch
import torchvision

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize, Grayscale, Pad
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU
from torch.optim import Adam

import nltk

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

import util, models

from tensorboardX import SummaryWriter

from layers import PlainMaskedConv2d, MaskedConv2d

SEEDFRAC = 2

def draw_sample(seeds, decoder, pixcnn, zs, seedsize=(0, 0)):

    b, c, h, w = seeds.size()

    sample = seeds.clone()
    if torch.cuda.is_available():
        sample, zs = sample.cuda(), zs.cuda()
    sample, zs = Variable(sample), Variable(zs)

    cond = decoder(zs)

    for i in tqdm.trange(h):
        for j in range(w):

            if i < seedsize[0] and j < seedsize[1]:
                continue

            for channel in range(c):

                result = pixcnn(sample, cond)
                probs = softmax(result[:, :, channel, i, j]).data

                pixel_sample = torch.multinomial(probs, 1).float() / 255.
                sample[:, channel, i, j] = pixel_sample.squeeze()

    return sample

def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    ## Load the data
    if arg.task == 'mnist':
        transform = Compose([Pad(padding=2), ToTensor()])

        trainset = torchvision.datasets.MNIST(root=arg.data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=arg.data_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 32, 32

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

        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir + os.sep + 'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir + os.sep + 'valid',
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

    OUTCN = 64

    if arg.model == 'vae-up':
        """
        Upsampling model. VAE with an encoder and a decoder, generates a conditional vector at every pixel,
        which is then passed to the picelCNN layers.
        """

        encoder = models.ImEncoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, colors=C)
        decoder = models.ImDecoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, out_channels=OUTCN)
        pixcnn  = models.LGated((C, H, W), OUTCN, arg.channels, num_layers=arg.num_layers, k=krn, padding=pad)

        mods = [encoder, decoder, pixcnn]

    elif arg.model == 'vae-straight':
        """
        Model that generates a single latent code for the whole image, and passes it straight to the autoregressive 
        decoder: no upsampling layers or deconvolutions.
        """

        encoder = models.ImEncoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, colors=C)
        decoder = util.Lambda(lambda x : x) # identity
        pixcnn  = models.CGated((C, H, W), (arg.zsize,), arg.channels, num_layers=arg.num_layers, k=krn, padding=pad)

        mods = [encoder, decoder, pixcnn]

    else:
        raise Exception('model "{}" not recognized'.format(arg.model))

    if torch.cuda.is_available():
        for m in mods:
            m.cuda()

    print('Constructed network', encoder, decoder, pixcnn)

    #
    sample_zs = torch.randn(12, arg.zsize)
    sample_zs = sample_zs.unsqueeze(1).expand(12, 6, -1).contiguous().view(72, 1, -1).squeeze(1)

    # A sample of 144 square images with 3 channels, of the chosen resolution
    # (144 so we can arrange them in a 12 by 12 grid)
    sample_init_zeros = torch.zeros(72, C, H, W)
    sample_init_seeds = torch.zeros(72, C, H, W)

    sh, sw = H//SEEDFRAC, W//SEEDFRAC

    # Init second half of sample with patches from test set, to seed the sampling
    testbatch = util.readn(testloader, n=12)
    testbatch = testbatch.unsqueeze(1).expand(12, 6, C, H, W).contiguous().view(72, 1, C, H, W).squeeze(1)
    sample_init_seeds[:, :, :sh, :] = testbatch[:, :, :sh, :]

    params = []
    for m in mods:
        params.extend(m.parameters())
    optimizer = Adam(params, lr=arg.lr)

    instances_seen = 0
    for epoch in range(arg.epochs):

        # Train
        err_tr = []

        for m in mods:
            m.train(True)

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
            zs = encoder(input)

            kl_loss = util.kl_loss(*zs)
            z = util.sample(*zs)

            out = decoder(z)

            rec = pixcnn(input, out)

            rec_loss = cross_entropy(rec, target, reduce=False).view(b, -1).sum(dim=1)
            rec_loss = rec_loss * util.LOG2E  # Convert from nats to bits

            loss = (rec_loss + kl_loss).mean()

            instances_seen += input.size(0)
            tbw.add_scalar('pixel-models/vae/training/kl-loss',  kl_loss.mean().data.item(), instances_seen)
            tbw.add_scalar('pixel-models/vae/training/rec-loss', rec_loss.mean().data.item(), instances_seen)

            err_tr.append(loss.data.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % arg.eval_every == 0 and epoch != 0:
            with torch.no_grad():

                # Evaluate
                # - we evaluate on the test set, since this is only a simple reproduction experiment
                #   make sure to split off a validation set if you want to tune hyperparameters for something important

                err_te = []

                for m in mods:
                    m.train(False)

                if not arg.skip_test:
                    for i, (input, _) in enumerate(tqdm.tqdm(testloader)):
                        if arg.limit is not None and i * arg.batch_size > arg.limit:
                            break

                        b, c, w, h = input.size()

                        if torch.cuda.is_available():
                            input = input.cuda()

                        target = (input.data * 255).long()
                        input, target = Variable(input), Variable(target)

                        zs = encoder(input)

                        kl_loss = util.kl_loss(*zs)
                        z = util.sample(*zs)

                        out = decoder(z)

                        rec = pixcnn(input, out)

                        rec_loss = cross_entropy(rec, target, reduce=False).view(b, -1).sum(dim=1)
                        rec_loss_bits = rec_loss * util.LOG2E  # Convert from nats to bits

                        loss = (rec_loss + kl_loss).mean()

                        err_te.append(loss.data.item())

                    tbw.add_scalar('pixel-models/test-loss', sum(err_te)/len(err_te), epoch)
                    print('epoch={:02}; training loss: {:.3f}; test loss: {:.3f}'.format(
                        epoch, sum(err_tr)/len(err_tr), sum(err_te)/len(err_te)))

                for m in mods:
                    m.train(False)

                sample_zeros = draw_sample(sample_init_zeros, decoder, pixcnn, sample_zs, seedsize=(0, 0))
                sample_seeds = draw_sample(sample_init_seeds, decoder, pixcnn, sample_zs, seedsize=(sh, W))
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
                        default='vae', type=str)

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

    parser.add_argument("--skip-test",
                        dest="skip_test",
                        help="Skips evaluation on the test set (but still takes a sample).",
                        action='store_true')

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("--evaluate-every",
                        dest="eval_every",
                        help="Run an evaluation/sample every n epochs.",
                        default=1, type=int)

    parser.add_argument("-k", "--kernel_size",
                        dest="kernel_size",
                        help="Size of convolution kernel",
                        default=7, type=int)

    parser.add_argument("-x", "--num-layers",
                        dest="num_layers",
                        help="Number of pixelCNN layers",
                        default=3, type=int)

    parser.add_argument("-d", "--vae-depth",
                        dest="vae_depth",
                        help="Depth of the VAE in blocks (in addition to the 3 default blocks). Each block halves the resolution in each dimension with a 2x2 maxpooling layer.",
                        default=0, type=int)

    parser.add_argument("-c", "--channels",
                        dest="channels",
                        help="Number of channels (aka feature maps) for the intermediate representations. Should be divisible by the number of colors.",
                        default=60, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-z", "--z-size",
                        dest="zsize",
                        help="Size of latent space.",
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