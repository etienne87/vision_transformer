"""
Tranformer models
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as f

from arch.unet import unet_layers, Unet
from core.conv import ResBlock, ConvLayer
from core.temporal import SequenceWise
from einops import rearrange


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, 1, padding, bias=False)

    def forward(self, *inputs):
        x, skip = inputs
        x = f.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, skip), dim=1)
        out = self.conv2d(x)
        return out



class UnetConv(nn.Module):
    """
    U-Net with Conv
    """
    def __init__(self, in_channels, out_channels, num_layers_enc, num_layers_dec, base=8, coords=True):
        super().__init__()
        base = 8
        downs = [base*2**(i+1) for i in range(num_layers_enc)]
        ups = [base*2**(num_layers_dec-i) for i in range(num_layers_dec)]

        down = lambda x,y:ResBlock(x,y,stride=2)
        midd = lambda x,y:ConvLayer(x,y)
        up = lambda x,y:UpConv(x,y)

        enc, dec = unet_layers(down, midd, up, base, downs, ups[0]*2, ups)
        self.unet = Unet(enc, dec)

        self.head = ConvLayer(in_channels + 2 * coords, base, 5, 1, 2, norm='BatchNorm2d')

        self.coords = coords
        self.predictor = nn.Conv2d(ups[-1], out_channels, 1, 1, 0)
        self.height, self.width = -1, -1

    def _generate_grid(self, height, width):
        self.height = height
        self.width = width
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1., 1., height), torch.linspace(-1., 1., width)])
        self.grid = torch.cat((grid_w[None, None, :, :], grid_h[None, None, :, :]), 1)

    def add_coords(self, y):
        height, width = y.shape[-2:]
        if [height, width] != [self.height, self.width]:
            self._generate_grid(height, width)
            self.grid.data = self.grid.data.type_as(y)
        grid = self.grid.expand(len(y),2,height,width)
        y = torch.cat((y, grid), dim=1)
        return y

    def forward(self, x):
        if self.coords:
            x = self.add_coords(x)
        y = self.head(x)
        y = self.unet(y)
        y = self.predictor(y)
        return y



if __name__ == '__main__':
    x = torch.randn(2,3,128,128)
    net = UnetConv(3, 11, 4, 4)
    y = net(x)
    print(y.shape)
