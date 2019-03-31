
import torch
from torch import nn
import torch.nn.functional as F

import util, sys

class PlainMaskedConv2d(nn.Conv2d):
    """
    Plain
    """
    def __init__(self, self_connection=False, *args, **kwargs):
        """
        This is the "vanilla" masked CNN. Note that this creates a blind spot in the receptive field when stacked.

        :param self_connection: Whether to mask out the "current pixel" (ie. the middle of the convolution). In the
         first layer, this should be masked out, since it connects to the value we're trying to predict. In higher layers
         it convery the intermediate representations we're building up.
        :param args: passed to the Conv2d layer
        :param kwargs: passed to the Conv2d layer
        """

        super(PlainMaskedConv2d, self).__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()

        self.mask.fill_(1)
        # zero the second half of the halfway row
        self.mask[:, :, kH // 2, kW // 2 + self_connection:] = 0
        # zero the bottom half rows
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(PlainMaskedConv2d, self).forward(x)

class MaskedConv2d(nn.Module):
    """
    Two-stack masked convolution with gated activations and a residual-connection

    See figure 2 in _Conditional Image Generation with PixelCNN Decoders_, van den Oord 2016.
    """
    def __init__(self, channels, colors, self_connection=False, res_connection=True, hv_connection=True, gates=True, k=7, padding=3):
        """
        This is the "vanilla" masked CNN. Note that this creates a blind spot in the receptive field when stacked.

        :param self_connection: Whether to mask out the "current pixel" (ie. the middle of the convolution). In the
         first layer, this should be masked out, since it connects to the value we're trying to predict. In higher layers
         it conveys the intermediate representations we're building up.
        :param channels: The number of channels of both the input and the output.
        :param conditional: Size of the value to condition on. None if no conditional
        """
        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k), padding=(0, padding), bias=False)
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1, padding=0, bias=False, groups=colors)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1, padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        # self.hmask[:, :, :, k // 2 + self_connection:] = 0
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)

        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        # print(self_connection + 0, self_connection, m)

        for c in range(0, colors):
            f, t = c * pc, (c+1) * pc

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1
                self.hmask[f+channels:t+channels, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f+pc, 0, m] = 1
                self.hmask[f + channels:t + channels, :f+pc, 0, m] = 1

        print(self.hmask[:, :, 0, m])

    def forward(self, x):

        vxin, hxin = x

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx =   self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = util.gate(vx)
            hx = util.gate(hx)

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, hx

class CMaskedConv2d(nn.Module):
    """
    Masked convolution, with location independent conditional.

    """
    def __init__(self, input_size, conditional_size, channels, colors=3, self_connection=False, res_connection=True, hv_connection=True, gates=True, k=7, padding=3):

        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k), padding=(0, padding), bias=False)
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1, padding=0, bias=False, groups=colors)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1, padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)

        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        # print(self_connection + 0, self_connection, m)

        for c in range(0, colors):
            f, t = c * pc, (c+1) * pc

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1
                self.hmask[f+channels:t+channels, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f+pc, 0, m] = 1
                self.hmask[f + channels:t + channels, :f+pc, 0, m] = 1

        print(self.hmask[:, :, 0, m])

        fr = util.prod(conditional_size)
        to = util.prod(input_size)

        # The conditional weights
        self.vhf = nn.Linear(fr, to)
        self.vhg = nn.Linear(fr, to)
        self.vvf = nn.Linear(fr, to)
        self.vvg = nn.Linear(fr, to)

    def forward(self, vxin, hxin, h):

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx =   self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = self.gate(vx, h, (self.vvf, self.vvg))
            hx = self.gate(hx, h, (self.vhf, self.vhg))

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, hx

    def gate(self, x, cond, weights):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gate activation.
        - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a mask
        - The two are element-wise multiplied, and the result is returned.

        Conditional and weights are used to compute a bias based on the conditional element

        :param x: The input tensor.
        :return: The input tensor x with the activation applied.
        """
        b, c, h, w = x.size()

        # compute conditional term
        vf, vg = weights

        tan_bias = vf(cond.view(b, -1)).view((b, c//2, h, w))
        sig_bias = vg(cond.view(b, -1)).view((b, c//2, h, w))

        # compute convolution term
        b = x.size(0)
        c = x.size(1)

        half = c // 2

        top = x[:, :half]
        bottom = x[:, half:]

        # apply gate and return
        return F.tanh(top + tan_bias) * F.sigmoid(bottom + sig_bias)


class LMaskedConv2d(nn.Module):
    """
    Masked convolution, with location dependent conditional.

    The conditional must be an 'image' tensor (BCHW) with the same resolution as the instance (no of channels can be different)

    """
    def __init__(self, input_size, conditional_channels, channels, colors=3, self_connection=False, res_connection=True, hv_connection=True, gates=True, k=7, padding=3):

        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k), padding=(0, padding), bias=False)
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1, padding=0, bias=False, groups=colors)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1, padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)

        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        # print(self_connection + 0, self_connection, m)

        for c in range(0, colors):
            f, t = c * pc, (c+1) * pc

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1
                self.hmask[f+channels:t+channels, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f+pc, 0, m] = 1
                self.hmask[f + channels:t + channels, :f+pc, 0, m] = 1

        print(self.hmask[:, :, 0, m])

        # The conditional weights
        self.vhf = nn.Conv2d(conditional_channels, channels, 1)
        self.vhg = nn.Conv2d(conditional_channels, channels, 1)
        self.vvf = nn.Conv2d(conditional_channels, channels, 1)
        self.vvg = nn.Conv2d(conditional_channels, channels, 1)

    def forward(self, vxin, hxin, h):

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx =   self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = self.gate(vx, h,  (self.vvf, self.vvg))
            hx = self.gate(hx, h,  (self.vhf, self.vhg))

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, hx

    def gate(self, x, cond, weights):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gate activation.
        - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a mask
        - The two are element-wise multiplied, and the result is returned.

        Conditional and weights are used to compute a bias based on the conditional element

        :param x: The input tensor.
        :return: The input tensor x with the activation applied.
        """
        b, c, h, w = x.size()

        # compute conditional term
        vf, vg = weights

        tan_bias = vf(cond)
        sig_bias = vg(cond)

        # compute convolution term
        b = x.size(0)
        c = x.size(1)

        half = c // 2

        top = x[:, :half]
        bottom = x[:, half:]

        # apply gate and return
        return F.tanh(top + tan_bias) * F.sigmoid(bottom + sig_bias)