"""
Tranformer models
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as f

from arch.unet import unet_layers, Unet 
from core.conv_rnn import ConvRNN
from core.conv import ResBlock, ConvLayer
from core.temporal import SequenceWise, seq_wise



class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, 1, padding, bias=False)

    def forward(self, *inputs):
        x, skip = inputs
        x = seq_wise(f.interpolate)(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat((x,skip), dim=2)
        out = seq_wise(self.conv2d)(x)
        return out


class UnetRNN(nn.Module):
    """
    U-Net with ConvRNN 
    """
    def __init__(self, in_channels, out_channels, num_layers=None):
        super().__init__()
        num_layers = 4
        base = 8 

        downs = [base*2**(i+1) for i in range(num_layers)]
        ups = [base*2**(num_layers-i) for i in range(num_layers)]


        down = lambda x,y:SequenceWise(ResBlock(x,y,stride=2))
        midd = lambda x,y:ConvRNN(x,y)
        up = lambda x,y:UpConv(x,y)

        enc, dec = unet_layers(down, midd, up, base, downs, ups[0]*2, ups)
        self.unet = Unet(enc, dec)

        self.head = SequenceWise(ConvLayer(in_channels, base, 5, 1, 2))
        self.predictor = SequenceWise(nn.Conv2d(ups[-1], out_channels, 1, 1, 0))

    def forward(self, x): 
        y = self.head(x) 
        y = self.unet(y)
        return self.predictor(y)

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        for module in self.unet.modules():
            if hasattr(module, "reset"):
                module.reset(mask)


