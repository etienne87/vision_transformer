"""
Transformer for Detection 
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

from core.transformer import Transformer
from core.positional_encoding import FixedPositionalEncoding, LearnedPositionalEncoding
from core.pooling import QuerySetAttention, SlotAttention
from einops import rearrange




class DetViT(nn.Module):
    def __init__(self, in_channels, out_channels, patch_dim=16, num_layers=2, num_heads=32, num_queries=16, embedding_dim=512, hidden_dim=512, max_len=512, dropout=0.):
        super().__init__()

        self.patch_dim = patch_dim

        self.flatten_dim_in = patch_dim * patch_dim * in_channels
        self.linear_encoding = nn.Linear(self.flatten_dim_in, embedding_dim)

        self.position_encoding = LearnedPositionalEncoding(max_len, embedding_dim)
        self.transformer = Transformer(embedding_dim, num_layers, num_heads, hidden_dim, dropout)

        self.pool = QuerySetAttention(num_queries, embedding_dim, num_heads) 
        #self.decoder = Transformer(embedding_dim, 1, num_heads, hidden_dim, dropout)


    def forward(self, x):
        b,c,h,w = x.shape
        p = self.patch_dim

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.linear_encoding(x)
        x = self.position_encoding(x)

        x = self.transformer(x)
        y = self.pool(x)
        return y

