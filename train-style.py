import os, tqdm, random, pickle, sys

import torch
import torchvision

from torch import nn
import torch.nn.functional as F
import torch.distributions as ds

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
DV = 'cuda' if torch.cuda.is_available() else 'cpu'

def standard(b, c, h, w):
    mean = torch.zeros(b, c, h, w)
    sig  = torch.ones(b, c, h, w)

    res = torch.cat([mean, sig], dim=1)

    if torch.cuda.is_available():
        res = res.cuda()
    return res

class StyleEncoder(nn.Module):

    def __init__(self, in_size, channels, zs=256, k=3):
        super().__init__()

        c, h, w = in_size
        c1, c2, c3, c4, c5 = channels

        # resnet blocks
        self.block1 = util.Block(c, c1, kernel_size=k)
        self.block2 = util.Block(c1, c2, kernel_size=k)
        self.block3 = util.Block(c2, c3, kernel_size=k)
        self.block4 = util.Block(c3, c4, kernel_size=k)
        self.block5 = util.Block(c4, c5, kernel_size=k)

        # affine mappings to distribution on latent space
        self.affine0 = nn.Linear(util.prod(in_size), 2 * zs)
        self.affine1 = nn.Linear(util.prod((c1, h//2, w//2)), 2 * zs)
        self.affine2 = nn.Linear(util.prod((c2, h//4, w//4)), 2 * zs)
        self.affine3 = nn.Linear(util.prod((c3, h//8, w//8)), 2 * zs)
        self.affine4 = nn.Linear(util.prod((c4, h//16, w//16)), 2 * zs)
        self.affine5 = nn.Linear(util.prod((c5, h//32, w//32)), 2 * zs)

        self.affinez = nn.Linear(12 * zs, 2 * zs)

        # 1x1 convolution to distribution on "noise space"
        self.tonoise0 = nn.Conv2d(c,  2*c,  kernel_size=1, padding=0)
        self.tonoise1 = nn.Conv2d(c1, 2*c1, kernel_size=1, padding=0)
        self.tonoise2 = nn.Conv2d(c2, 2*c2, kernel_size=1, padding=0)
        self.tonoise3 = nn.Conv2d(c3, 2*c3, kernel_size=1, padding=0)
        self.tonoise4 = nn.Conv2d(c4, 2*c4, kernel_size=1, padding=0)
        self.tonoise5 = nn.Conv2d(c5, 2*c5, kernel_size=1, padding=0)

        self.unmapping = nn.Sequential(
            nn.Linear(zs*2, zs*2), nn.ReLU(),
            nn.Linear(zs*2, zs*2), nn.ReLU(),
            nn.Linear(zs*2, zs*2), nn.ReLU(),
            nn.Linear(zs*2, zs*2), nn.ReLU(),
            nn.Linear(zs*2, zs*2), nn.ReLU(),
            nn.Linear(zs*2, zs*2), nn.ReLU(),
            nn.Linear(zs*2, zs*2), nn.ReLU(),
            nn.Linear(zs*2, zs*2), nn.ReLU()
        )

    def forward(self, x0):

        b = x0.size(0)

        z0 = self.affine0(x0.view(b, -1))
        n0 = self.tonoise0(x0)

        x1 = F.max_pool2d(self.block1(x0), 2)
        z1 = self.affine1(x1.view(b, -1))
        n1 = self.tonoise1(x1)

        x2 = F.max_pool2d(self.block2(x1), 2)
        z2 = self.affine2(x2.view(b, -1))
        n2 = self.tonoise2(x2)

        x3 = F.max_pool2d(self.block3(x2), 2)
        z3 = self.affine3(x3.view(b, -1))
        n3 = self.tonoise3(x3)

        x4 = F.max_pool2d(self.block4(x3), 2)
        z4 = self.affine4(x4.view(b, -1))
        n4 = self.tonoise4(x4)

        x5 = F.max_pool2d(self.block5(x4), 2)
        z5 = self.affine5(x5.view(b, -1))
        n5 = self.tonoise5(x5)

        # average the z vectors
        zbatch = torch.cat([
            z0[:, :, None],
            z1[:, :, None],
            z2[:, :, None],
            z3[:, :, None],
            z4[:, :, None],
            z5[:, :, None]], dim=2)
        z = self.affinez(zbatch.view(b, -1))

        z = self.unmapping(z)

        return z, n0, n1, n2, n3, n4, n5

class StyleDecoder(nn.Module):

    def __init__(self, out_size, channels, zs=256, k=3, dist='gaussian'):
        super().__init__()

        self.out_size = out_size

        c, h, w = out_size
        c1, c2, c3, c4, c5 = channels

        # resnet blocks
        self.block5 = util.Block(c5*2, c4, kernel_size=k, deconv=True)
        self.block4 = util.Block(c4*3, c3, kernel_size=k, deconv=True)
        self.block3 = util.Block(c3*3, c2, kernel_size=k, deconv=True)
        self.block2 = util.Block(c2*3, c1, kernel_size=k, deconv=True)
        self.block1 = util.Block(c1*3, c,  kernel_size=k, deconv=True)

        # affine mappings from latent space sample
        self.affine5 = nn.Linear(zs, util.prod((c5, h//32, w//32)))
        self.affine4 = nn.Linear(zs, util.prod((c4, h//16, w//16)))
        self.affine3 = nn.Linear(zs, util.prod((c3, h//8, w//8)))
        self.affine2 = nn.Linear(zs, util.prod((c2, h//4, w//4)))
        self.affine1 = nn.Linear(zs, util.prod((c1, h//2, w//2)))
        self.affine0 = nn.Linear(zs, util.prod(out_size))

        # 1x1 convolution from "noise space" sample
        self.tonoise5 = nn.Conv2d(c5, 2*c5, kernel_size=1, padding=0)
        self.tonoise4 = nn.Conv2d(c4, 2*c4, kernel_size=1, padding=0)
        self.tonoise3 = nn.Conv2d(c3, 2*c3, kernel_size=1, padding=0)
        self.tonoise2 = nn.Conv2d(c2, 2*c2, kernel_size=1, padding=0)
        self.tonoise1 = nn.Conv2d(c1, 2*c1, kernel_size=1, padding=0)
        self.tonoise0 = nn.Conv2d(c,  2*c,  kernel_size=1, padding=0)

        # mapping to distribution on image space
        if dist in ['gaussian','beta']:
            self.conv0 = nn.Conv2d(c*3, c*2, kernel_size=1)
        elif dist == 'bernoulli':
            self.conv0 = nn.Conv2d(c * 3, c, kernel_size=1)
        else:
            raise Exception('Output distribution {} not recognized'.format(dist))

        self.mapping = nn.Sequential(
            nn.Linear(zs, zs), nn.ReLU(),
            nn.Linear(zs, zs), nn.ReLU(),
            nn.Linear(zs, zs), nn.ReLU(),
            nn.Linear(zs, zs), nn.ReLU(),
            nn.Linear(zs, zs), nn.ReLU(),
            nn.Linear(zs, zs), nn.ReLU(),
            nn.Linear(zs, zs), nn.ReLU(),
            nn.Linear(zs, zs), nn.ReLU()
        )

    def forward(self, z, n0, n1, n2, n3, n4, n5):

        z = self.mapping(z)

        z5 = self.affine5(z).view(*n5.size())
        x5 = torch.cat([z5, n5], dim=1)

        x4 = F.upsample_bilinear(self.block5(x5), scale_factor=2)
        z4 = self.affine4(z).view(*n4.size())
        x4 = torch.cat([z4, n4, x4], dim=1)

        x3 = F.upsample_bilinear(self.block4(x4), scale_factor=2)
        z3 = self.affine3(z).view(*n3.size())
        x3 = torch.cat([z3, n3, x3], dim=1)

        x2 = F.upsample_bilinear(self.block3(x3), scale_factor=2)
        z2 = self.affine2(z).view(*n2.size())
        x2 = torch.cat([z2, n2, x2], dim=1)

        x1 = F.upsample_bilinear(self.block2(x2), scale_factor=2)
        z1 = self.affine1(z).view(*n1.size())
        x1 = torch.cat([z1, n1, x1], dim=1)

        x0 = F.upsample_bilinear(self.block1(x1), scale_factor=2)
        z0 = self.affine0(z).view(*n0.size())
        x0 = torch.cat([z0, n0, x0], dim=1)

        x0 = self.conv0(x0)

        return torch.sigmoid(x0)

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

    elif arg.task == 'ffhq':
        transform = Compose([ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 128, 128

    else:
        raise Exception('Task {} not recognized.'.format(arg.task))

    channels = (8, 16, 32, 64, 128)

    zs = arg.latent_size

    encoder = StyleEncoder((C, H, W), channels, zs=zs, k=arg.kernel_size)
    decoder = StyleDecoder((C, H, W), channels, zs=zs, k=arg.kernel_size)

    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=arg.lr)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    instances_seen = 0
    for epoch in range(arg.epochs):

        # Train
        err_tr = []
        encoder.train(True)
        decoder.train(True)

        for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

            # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()

            input, target = Variable(input), Variable(input)

            # -- encoding
            z, n0, n1, n2, n3, n4, n5 = encoder(input)

            # -- compute KL losses

            zkl  = util.kl_loss(z[:, :zs], z[:, zs:])
            n0kl = util.kl_loss_image(n0)
            n1kl = util.kl_loss_image(n1)
            n2kl = util.kl_loss_image(n2)
            n3kl = util.kl_loss_image(n3)
            n4kl = util.kl_loss_image(n4)
            n5kl = util.kl_loss_image(n5)

            # -- take samples
            zsample  = util.sample(z[:, :zs], z[:, zs:])
            n0sample = util.sample_image(n0)
            n1sample = util.sample_image(n1)
            n2sample = util.sample_image(n2)
            n3sample = util.sample_image(n3)
            n4sample = util.sample_image(n4)
            n5sample = util.sample_image(n5)

            # -- decoding
            xout = decoder(zsample, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample)

            # Only Gaussian loss for now
            m = ds.Normal(xout[:, :C, :, :], xout[:, C:, :, :])
            rec_loss = - m.log_prob(target).sum(dim=1).sum(dim=1).sum(dim=1)

            br, bz, b0, b1, b2, b3, b4, b5 = arg.betas

            loss = br * rec_loss + bz * zkl + b0 * n0kl + b1 * n1kl + b2 * n2kl + b3 * n3kl + b4 * n4kl + b5 * n5kl
            loss = loss.mean(dim=0)

            instances_seen += input.size(0)

            tbw.add_scalar('style-vae/zkl-loss', float(zkl.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/n0kl-loss', float(n0kl.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/n1kl-loss', float(n1kl.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/n2kl-loss', float(n2kl.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/n3kl-loss', float(n3kl.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/n4kl-loss', float(n4kl.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/n5kl-loss', float(n5kl.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/rec-loss', float(rec_loss.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/total-loss', float(loss.data.item()), instances_seen)

            # Backward pass
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        err_te = []
        encoder.train(False)
        decoder.train(False)

        if not arg.skip_test:
            for i, (input, _) in enumerate(tqdm.tqdm(testloader)):
                if arg.limit is not None and i * arg.batch_size > arg.limit:
                    break

                if torch.cuda.is_available():
                    input = input.cuda()

                input, target = Variable(input), Variable(input)

                # -- encoding
                z, n0, n1, n2, n3 = encoder(input)

                # -- compute KL losses

                zkl  = util.kl_loss(z[:, :zs], z[:, zs:])
                n0kl = util.kl_loss_image(n0)
                n1kl = util.kl_loss_image(n1)
                n2kl = util.kl_loss_image(n2)
                n3kl = util.kl_loss_image(n3)
                n4kl = util.kl_loss_image(n4)
                n5kl = util.kl_loss_image(n5)

                # -- take samples
                zsample  = util.sample(z[:, :zs], z[:, zs:])
                n0sample = util.sample_image(n0)
                n1sample = util.sample_image(n1)
                n2sample = util.sample_image(n2)
                n3sample = util.sample_image(n3)
                n4sample = util.sample_image(n4)
                n5sample = util.sample_image(n5)

                # -- decoding
                xout = decoder(zsample, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample)

                # Only Gaussian loss for now
                m = ds.Normal(xout[:, :C, :, :], xout[:, C:, :, :])
                rec_loss = -m.log_prob(target).sum(dim=1).sum(dim=1).sum(dim=1)

                loss = rec_loss + zkl + n0kl + n1kl + n2kl + n3kl + n4kl + n5kl
                loss = loss.mean(dim=0)

                err_te.append(loss.data.item())

            tbw.add_scalar('pixel-models/test-loss', sum(err_te)/len(err_te), epoch)
            print('epoch={:02}; test loss: {:.3f}'.format(
                epoch, sum(err_te)/len(err_te)))

        # take some samples

        # sample 6x12 images
        b = 6 * 12

        zrand  = torch.randn(b, zs, device=DV)
        n0rand = torch.randn(b, C, H, W, device=DV)
        n1rand = torch.randn(b, channels[0], H//2, W//2, device=DV)
        n2rand = torch.randn(b, channels[1], H//4, W//4, device=DV)
        n3rand = torch.randn(b, channels[2], H//8, W//8, device=DV)
        n4rand = torch.randn(b, channels[3], H//16, W//16, device=DV)
        n5rand = torch.randn(b, channels[4], H//32, W//32, device=DV)

        sample = decoder(zrand, n0rand, n1rand, n2rand, n3rand, n4rand, n5rand).clamp(0, 1)[:, :C, :, :]

        # reconstruct 6x12 images from the testset
        input = util.readn(testloader, n=6*12)
        if torch.cuda.is_available():
            input = input.cuda()
        input = Variable(input)

        # -- encoding
        z, n0, n1, n2, n3, n4, n5 = encoder(input)

        # -- take samples
        zsample = util.sample(z[:, :zs], z[:, zs:])
        n0sample = util.sample_image(n0)
        n1sample = util.sample_image(n1)
        n2sample = util.sample_image(n2)
        n3sample = util.sample_image(n3)
        n4sample = util.sample_image(n4)
        n5sample = util.sample_image(n5)

        # -- decoding
        xout = decoder(zsample, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample).clamp(0, 1)[:, :C, :, :]

        # -- mix the latent vector with random noise
        mixout = decoder(zsample, n0rand, n1rand, n2rand, n3rand, n4rand, n5rand).clamp(0, 1)[:, :C, :, :]

        # -- mix the a random vector with the sample noise
        mixout2 = decoder(zrand, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample).clamp(0, 1)[:, :C, :, :]

        images = torch.cat([sample, input, xout, mixout, mixout2], dim=0)

        utils.save_image(images, 'images_{:02d}.png'.format(epoch), nrow=24, padding=2)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task: [mnist, cifar10].",
                        default='mnist', type=str)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("--skip-test",
                        dest="skip_test",
                        help="Skips evaluation on the test set (but still takes a sample).",
                        action='store_true')

    parser.add_argument("--evaluate-every",
                        dest="eval_every",
                        help="Run an exaluation/sample every n epochs.",
                        default=1, type=int)

    parser.add_argument("-k", "--kernel_size",
                        dest="kernel_size",
                        help="Size of convolution kernel",
                        default=3, type=int)
    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-z", "--latent-size",
                        dest="latent_size",
                        help="Size of latent space.",
                        default=128, type=int)

    parser.add_argument('--betas',
                        dest='betas',
                        help="Scaling parameters of the kl losses. The first two are for reconstruction loss and the z parameter, the rest are for the noise parameters in order. Provide exactly 7 floats.",
                        nargs=8,
                        type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

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
                        default='./runs/style', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)