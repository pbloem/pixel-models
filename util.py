import string, re

import numpy as np
import os, sys, math
import datetime, pathlib
import random

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

LOG2E = math.log2(math.e)

def log_anneal(step, total, temp=1):
    """ Logistic annealing function
    :param step:
    :param total:
    :param temp: Steepness of the curves
    :return:
    """
    return float(1 / (1 + np.exp(-temp * (step*2 - total))))

def lin_anneal(step, total, ceiling=0.5):
    """ Linear annealing function
    :param step:
    :param total:
    :param ceiling: Where in the process the aneal should hit 1. If 0.5, the anneal linearly climbs upt to 1 and then
        hits 1 halfway
    :return:
    """
    return min(1, (step/ceiling) / total)

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    axes.spines["right"].set_visible(False)
    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["left"].set_visible(False)

    axes.get_xaxis().set_tick_params(which='both', top='off', bottom='off', labeltop='off', labelbottom='off')
    axes.get_yaxis().set_tick_params(which='both', left='off', right='off', labelleft='off', labelright='off')

def kl_loss(zmean, zlsig):

    b, l = zmean.size()

    kl = 0.5 * torch.sum(zlsig.exp() - zlsig + zmean.pow(2) - 1, dim=1)

    assert kl.size() == (b,)

    return kl

def sample(zmean, zlsig, eps=None):
    b, l = zmean.size()

    if eps is None:
        eps = torch.randn(b, l)
        if torch.cuda.is_available():
            eps = eps.cuda()
        eps = Variable(eps)

    return zmean + eps * (zlsig * 0.5).exp()

def kl_loss_image(z):

    b, c, h, w = z.size()

    mean = z[:, :c//2, :, :].view(b, -1)
    sig = z[:, c//2:, :, :].view(b, -1)

    kl = 0.5 * torch.sum(sig.exp() - sig + mean.pow(2) - 1, dim=1)

    assert kl.size() == (b,)

    return kl

def sample_image(z, eps=None):

    b, c, h, w = z.size()

    mean = z[:, :c//2, :, :].view(b, -1)
    sig = z[:, c//2:, :, :].view(b, -1)

    if eps is None:
        eps = torch.randn(b, c//2, h, w).view(b, -1)
        if torch.cuda.is_available():
            eps = eps.cuda()
        eps = Variable(eps)

    sample = mean + eps * (sig * 0.5).exp()

    return sample.view(b, c//2, h, w)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def ensure(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

REGEX = re.compile('[%s]' % re.escape(string.punctuation))

def tokenize(sentence : str):
    """Simple and fast tokenization."""

    return REGEX.sub('', sentence.lower()).split()

def intseq(words, w2i, unk='.unk'):
    """
    Convert a word sequence to an integer sequence based on the given codebook.
    :param words:
    :param w2i:
    :param unk:
    :return:
    """
    res = [None] * len(words)
    for j, word in enumerate(words):
            if word in w2i:
                res[j] = w2i[word]
            else:
                res[j] = w2i[unk]

    return res

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Debug(nn.Module):
    """
    Executes a lambda function and then returns the input. Useful for debugging.
    """
    def __init__(self, lambd):
        super(Debug, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        self.lambd(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape)


class Block(nn.Module):

    def __init__(self, in_channels, channels, num_convs = 3, kernel_size = 3, batch_norm=False, use_weight=True, use_res=True, deconv=False):
        super().__init__()

        layers = []
        self.use_weight = use_weight
        self.use_res = use_res

        padding = int(math.floor(kernel_size / 2))

        self.upchannels = nn.Conv2d(in_channels, channels, kernel_size=1)

        for i in range(num_convs):
            if deconv:
                layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))
            else:
                layers.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))

            if batch_norm:
                layers.append(module=nn.BatchNorm2d(channels))

            layers.append(nn.ReLU())

        self.seq = nn.Sequential(*layers)

        if use_weight:
            self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):

        x = self.upchannels(x)

        out = self.seq(x)

        if not self.use_res:
            return out

        if not self.use_weight:
            return out + x

        return out + self.weight * x

def interpolate(images, encoder, decoder, steps=7, name='interpolate', mode='spherical', reps=5):
    """
    Plots a grid of values interpolating (linearly)  between four given items.

    :param name:
    :return:
    """

    plt.figure(figsize=(steps + 2, reps))

    f, aa = plt.subplots(reps, steps + 2, gridspec_kw = {'wspace':0, 'hspace':0.01})

    for rep in range(reps):

        x1 = images[random.randint(0, images.size(0))]
        x2 = images[random.randint(0, images.size(0))]

        x1, x2 = x1.unsqueeze(0).float(), x2.unsqueeze(0).float()

        if torch.cuda.is_available():
            x1, x2 = x1.cuda(), x2.cuda()
        x1, x2 = Variable(x1), Variable(x2)

        z1, z2 = encoder(x1), encoder(x2)

        if mode == 'spherical':
            zs = slerp(z1, z2, steps)
        elif mode == 'linear':
            zs = linp(z1, z2, steps)
        else:
            raise Exception('Mode {} not recognized'.format(mode))

        out = decoder(zs).data
        out = np.transpose(out.cpu().numpy(), (0, 2, 3, 1))

        for i in range(steps):
            aa[rep, i+1].imshow(out[i])

        aa[rep,  0].imshow(np.transpose(x1[0].cpu().numpy(), (1, 2, 0)))
        aa[rep, -1].imshow(np.transpose(x2[0].cpu().numpy(), (1, 2, 0)))

    for i in range(aa.shape[0]):
        for j in range(aa.shape[1]):
            clean(aa[i,j])

    plt.savefig(name + '.pdf')

def interpolate(zpairs, decoder, steps=9, name='interpolate', mode='spherical', reps=5):
    """
    Plots a grid of values interpolating (linearly)  between four given items.

    :param name:
    :return:
    """

    plt.figure(figsize=(steps, len(zpairs)))

    f, aa = plt.subplots(len(zpairs), steps, gridspec_kw = {'wspace':0, 'hspace':0.01})

    for r, zpair in enumerate(zpairs):

        z1, z2 = zpair

        if mode == 'spherical':
            zs = slerp(z1, z2, steps)
        elif mode == 'linear':
            zs = linp(z1, z2, steps)
        else:
            raise Exception('Mode {} not recognized'.format(mode))

        out = decoder(zs).data
        out = np.transpose(out.cpu().numpy(), (0, 2, 3, 1))

        for i in range(steps):
            aa[r, i].imshow(out[i])

    for i in range(aa.shape[0]):
        for j in range(aa.shape[1]):
            clean(aa[i,j])

    plt.savefig(name + '.pdf')

def linp(x, y, steps=5):
    """
    Produces a spherical linear interpolation between two points

    :param x:
    :param y:
    :param steps:
    :return:
    """
    assert x.size(0) == y.size(0)
    n = x.size(0)

    d = torch.linspace(0, 1, steps)

    return x.unsqueeze(0).expand(steps, n) * d.unsqueeze(1) \
           + y.unsqueeze(0).expand(steps, n) * (1-d).unsqueeze(1)

def slerp(x, y, steps=5):
    """
    Produces a spherical linear interpolation between two points

    :param x: 1 by n matrix or length-n vector
    :param y:
    :param steps:
    :return:
    """
    assert x.size(0) == y.size(0)

    if len(x.size()) > 1:
        x, y = x[0], y[0]

    n = x.size(0)

    angle = torch.acos(torch.dot(x, y)/(x.norm() * y.norm()))

    d = torch.linspace(0, 1, steps).unsqueeze(1)

    if torch.cuda.is_available():
        d     = d.cuda()
        angle = angle.cuda()

    d1 = torch.sin((1-d) * angle) / torch.sin(angle)
    d2 = torch.sin(d     * angle) / torch.sin(angle)

    return   x.unsqueeze(0).expand(steps, n) * d1 \
           + y.unsqueeze(0).expand(steps, n) * d2


def pad(sequences):

    bsize = len(sequences)
    lengths = [len(s) for s in sequences]

    result = torch.zeros(bsize, max(lengths), dtype=torch.long)

    for i, seq in enumerate(sequences):
        for j, val in enumerate(seq):
            result[i, j] = val

    return result, lengths

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def sample_logits(logits, temperature=1.0):

    """
    Sample an index from a (batched) logit vector.

    :param preds:
    :param temperature:
    :return:
    """
    b, v = logits.size()

    if temperature == 0.0:
        return torch.argmax(logits, dim= 1)

    preds = logits / temperature
    preds = preds - logsumexp(preds, dim=1, keepdim=True)

    choice = torch.multinomial(torch.exp(preds), num_samples=1)

    return choice.view(-1)

def logsumexp(x, dim=None, keepdim=False):
    # https://github.com/pytorch/pytorch/issues/2591

    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)

def gate(x):
    """
    Takes a batch x channels x rest... tensor and applies an LTSM-style gated activation.
    - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
    - The bottom half are fed through a sigmoid, functioning as a mask
    - The two are element-wise multiplied, and the result is returned.

    :param x: The input tensor.
    :return: The input tensor x with the activation applied.
    """
    b = x.size(0)
    c = x.size(1)

    half = c // 2

    top    = x[:, :half]
    bottom = x[:, half:]

    return F.tanh(top) * F.sigmoid(bottom)

def readn(loader, n, cls=False, maxval=None):
    """
    Reads from the loader to fill a large batch of size n
    :param loader: a dataloader
    :param n:
    :return:
    """

    batches = []
    total = 0
    for input in loader:
        batch = input[0] if not cls else input[1]

        if cls:
            batch = one_hot(batch, maxval)

        total += batch.size(0)
        batches.append(batch)

        if total > n:
            break

    result = torch.cat(batches, dim=0)

    return result[:n]


def one_hot(integers, maxval):
    """
    Converts a list of integer values to a one hot coded batch
    :param integers:
    :param maxval:
    :return:
    """

    FT = torch.cuda.FloatTensor if integers.is_cuda else torch.FloatTensor

    result = FT(integers.size(0), maxval).zero_()
    result.scatter_(dim=1, index=integers.unsqueeze(1), value=1)

    return result


def prod(xs):
    res = 1

    for x in xs:
        res *= x

    return res

def batched(input, model, batch_size, cuda=torch.cuda.is_available()):
    """
    Performs inference in batches. Input and output are non-variable, non-gpu tensors.

    :param input:
    :param model:
    :param batch_size:
    :param cuda:
    :return:
    """
    n = input.size(0)

    out_batches = []
    for fr in range(0, n, batch_size):
        to = min(n, fr + batch_size)

        batch = input[fr:to]
        if cuda:
            batch = batch.cuda()
        batch = Variable(batch)

        out_batches.append(model(batch).cpu().data)

        del batch

    return torch.cat(out_batches, dim=0)


