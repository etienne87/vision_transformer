"""basic cnn, no transformer
   reshape to 1d at the end
"""
import torch
import torch.nn as nn
from core.conv import ConvLayer
from einops import rearrange


class CNN4(nn.Module):
    def __init__(self, in_channels, out_channels, base=32, num_layers=0):
        super().__init__()
        self.channels = [in_channels] + [base * 2**i for i in range(4)]
        self.layers = nn.Sequential(
                *[ConvLayer(self.channels[i], self.channels[i+1], 5, 2, 2) for i in range(4)]
        ) 
        self.linear_decoding = nn.Linear(self.layers[-1].out_channels, out_channels)

    def forward(self, x):
        x = self.layers(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.linear_decoding(x)
        return x
