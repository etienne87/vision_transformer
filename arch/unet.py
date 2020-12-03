"""
Base unet code
U-Net: Convolutional Networks for Biomedical Image Segmentation
Olaf Ronneberger, Philipp Fischer, Thomas Brox

User is responsible for making sure spatial sizes agree.
"""
import torch
import torch.nn as nn



def unet_layers(down_block, middle_block, up_block, input_size=5, down_filter_sizes=[32,64], middle_filter_size=128, up_filter_sizes=[64,32,8]):
    """Builds unet layers 
    can be used to build unet layers (but you are not forced to)

    Args:
        down_block: encoder's block type
        middle_block: bottleneck's block type
        up_block: decoder's block type
        input_size: in_channels
        down_filter_sizes: out_channels per encoder
        middle_filter_size: bottleneck's channels
        up_fitler_sizes: decoder's channels
    """
    encoders = []
    encoders_channels = [input_size]
    last_channels = input_size
    for cout in down_filter_sizes:
        enc = down_block(encoders_channels[-1], cout)
        encoders.append(enc)
        encoders_channels.append(cout)

    middle = middle_block(encoders_channels[-1], middle_filter_size)
    decoders = [middle]
    decoders_channels = [middle_filter_size]
    for i, cout in enumerate(up_filter_sizes):
        cin = decoders_channels[-1] + encoders_channels[-i-2] #note index = -2! last encoder is at the same scale of current input, this is not what we want!
        decoders.append(up_block(cin, cout))
        decoders_channels.append(cout)

    return encoders, decoders 



class Unet(nn.Module):
    """Ultra-Generic Unet

    Args:
        encoders: list of encoder layers
        decoders: list of decoder layers
    """
    def __init__(self, encoders, decoders):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

    def forward(self, x):
        enc = [x]
        for down_layer in self.encoders:
            x = down_layer(x)
            enc.append(x)

        dec = [self.decoders[0](x)]
        for i, up_layer in enumerate(self.decoders[1:]):
            skip = enc[-i-2]
            x = up_layer(dec[-1], skip)
            dec.append(x)

        return dec[-1]
