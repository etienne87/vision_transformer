import torch
import torch.nn as nn
import torch.nn.functional as f


class DepthWiseSeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 depth_multiplier=1,
                 **kwargs):
        super(DepthWiseSeparableConv2d, self).__init__(
            nn.Conv2d(in_channels, int(in_channels * depth_multiplier), kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=in_channels, bias=bias, **kwargs),
            nn.Conv2d(int(depth_multiplier * in_channels), out_channels, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=bias)
        )


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=True, norm="BatchNorm2d", activation='ReLU', separable=False, **kwargs):

        conv_func = DepthWiseSeparableConv2d if separable else nn.Conv2d
        self.out_channels = out_channels
        self.separable = separable
        if not separable and "depth_multiplier" in kwargs:
            kwargs.pop('depth_multiplier')

        normalizer = nn.Identity() if norm == 'none' else getattr(nn, norm)(in_channels)

        super(ConvLayer, self).__init__(
            normalizer,
            conv_func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=padding, bias=bias, **kwargs),
            getattr(nn, activation)()
        )
        self.in_channels = in_channels
        self.out_channels = out_channels


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm="BatchNorm2d"):
        super(ResBlock, self).__init__()
        bias = norm == 'none'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            norm=norm,
        )
        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm=norm,
            bias=False,
        )

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
                norm=norm,
                bias=False,
                activation="Identity",
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.downsample(x)
        out = f.relu(out)
        return out
