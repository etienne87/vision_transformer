"""
Linear Decoding for pixel prediction
"""
import torch
import torch.nn as nn


class PatchDecoding(nn.Module):
    def __init__(self, out_channels, patch_dim=16): 
        super().__init__()
        self.flatten_dim_out = patch_dim * patch_dim * out_channels
        self.linear_decoding = nn.Linear(embedding_dim, self.flatten_dim_out)

    def forward(self, x):
        x = self.linear_decoding(x)
        l1 =  h // self.patch_dim
        l2 =  w // self.patch_dim
        y = rearrange(x, 'b (l1 l2) (p1 p2 d) -> b d (l1 p1) (l2 p2)', l1=l1, l2=l2, p1=self.patch_dim, p2=self.patch_dim)
        return y
