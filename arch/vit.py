"""
Transformer for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.transformer import Transformer
from core.positional_encoding import FixedPositionalEncoding, LearnedPositionalEncoding
from einops import rearrange




class ViT(nn.Module):
    def __init__(self, in_channels, out_channels, patch_dim=16, num_layers=2, num_heads=32, embedding_dim=512, hidden_dim=512, max_len=512, dropout=0.):
        super().__init__()

        self.patch_dim = patch_dim 
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim

        self.flatten_dim_in = patch_dim * patch_dim * in_channels
        self.linear_encoding = nn.Linear(self.flatten_dim_in, embedding_dim)

        self.position_encoding = LearnedPositionalEncoding(max_len, embedding_dim)

        self.transformer = Transformer(embedding_dim, num_layers, num_heads, hidden_dim, dropout)

        self.flatten_dim_out = patch_dim * patch_dim * out_channels
        self.linear_decoding = nn.Linear(embedding_dim, self.flatten_dim_out)

    def forward(self, x):
        b,c,h,w = x.shape
        p = self.patch_dim
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)

        x = self.linear_encoding(x)
        x = self.position_encoding(x)

        x = self.transformer(x)

        x = self.linear_decoding(x)
        l1 =  h // self.patch_dim
        l2 =  w // self.patch_dim
        y = rearrange(x, 'b (l1 l2) (p1 p2 d) -> b d (l1 p1) (l2 p2)', l1=l1, l2=l2, p1=self.patch_dim, p2=self.patch_dim)
        return y




