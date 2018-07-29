
import torch
from torch import nn

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

