"""
Transformer for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.transformer import Transformer
from core.positional_encoding import FixedPositionalEncoding, LearnedPositionalEncoding
from einops import rearrange


class ViT3d(nn.Module):
    def __init__(self, in_channels, out_channels, patch_dim=[16,16,1], num_layers=2, num_heads=32, embedding_dim=512, hidden_dim=512, max_len=4096, dropout=0.):
        super().__init__()

        self.patch_dim = patch_dim 
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.patch_len = patch_dim[0]*patch_dim[1]*patch_dim[2]

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.flatten_dim_in = self.patch_len * in_channels
        self.linear_encoding = nn.Linear(self.flatten_dim_in, embedding_dim)

        self.position_encoding = LearnedPositionalEncoding(max_len, embedding_dim)

        self.transformer = Transformer(embedding_dim, num_layers, num_heads, hidden_dim, dropout)

        self.flatten_dim_out = self.patch_len * out_channels
        self.linear_decoding = nn.Linear(embedding_dim, self.flatten_dim_out)

    def forward(self, x):
        t,b,c,h,w = x.shape
        x = x.permute(1,2,3,4,0).contiguous()

        p1, p2, p3 = self.patch_dim
        x = rearrange(x, 'b c (h p1) (w p2) (z p3) -> b (h w z) (p1 p2 p3 c)', p1 = p1, p2 = p2, p3=p3)

        x = self.linear_encoding(x)
        x = self.position_encoding(x)

        x = self.transformer(x)

        x = self.linear_decoding(x)
        l1 =  h // p1 
        l2 =  w // p2 
        l3 =  t // p3
        y = rearrange(x, 'b (l1 l2 l3) (p1 p2 p3 d) -> b d (l1 p1) (l2 p2) (l3 p3)', l1=l1, l2=l2, l3=l3, p1=p1, p2=p2, p3=p3)
        
        y = y.permute(4,0,1,2,3).contiguous()
        return y


