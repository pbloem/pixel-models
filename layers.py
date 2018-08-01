
import torch
from torch import nn

import util

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
    def __init__(self, channels, self_connection=False, res_connection=True, hv_connection=True, gates=True, k=7, padding=3):
        """
        This is the "vanilla" masked CNN. Note that this creates a blind spot in the receptive field when stacked.

        :param self_connection: Whether to mask out the "current pixel" (ie. the middle of the convolution). In the
         first layer, this should be masked out, since it connects to the value we're trying to predict. In higher layers
         it conveys the intermediate representations we're building up.
        :param channels: The number of channels of both the input and the output.
        """
        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        # TODO: should these have biases?
        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k, padding=padding)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k), padding=(0, padding))
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1, padding=0)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1, padding=0)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + self_connection:] = 0

    def forward(self, x):

        vx, hx, = x

        hx_in = hx

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx =   self.vertical.forward(vx)
        hx = self.horizontal.forward(hx)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = util.gate(vx)
            hx = util.gate(hx)

        if self.res_connection:
            hx = self.tores(hx) + hx_in

        return vx, hx

