"""
Transformer for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

from core.transformer import Transformer
from core.positional_encoding import FixedPositionalEncoding, LearnedPositionalEncoding
from core.attention_rnn import AttentionRNN
from einops import rearrange




class ViTRNN(nn.Module):
    def __init__(self, in_channels, out_channels, patch_dim=16, num_layers=2, memory_tokens=32, num_heads=32, embedding_dim=512, hidden_dim=512, max_len=512, dropout=0., conv_representation=False):
        super().__init__()

        self.patch_dim = patch_dim

        self.flatten_dim_in = patch_dim * patch_dim * in_channels
        self.linear_encoding = nn.Linear(self.flatten_dim_in, embedding_dim)

        self.transformer = AttentionRNN(embedding_dim, embedding_dim, memory_tokens, num_layers, num_heads, hidden_dim, max_len, dropout)

        self.flatten_dim_out = patch_dim * patch_dim * out_channels
        self.linear_decoding = nn.Linear(embedding_dim, self.flatten_dim_out)

    def forward(self, x):
        t,b,c,h,w = x.shape
        p = self.patch_dim

        x = rearrange(x, 't b c (h p1) (w p2) -> t b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.linear_encoding(x)

        x = self.transformer(x)

        x = self.linear_decoding(x)
        l1 =  h // self.patch_dim
        l2 =  w // self.patch_dim
        y = rearrange(x, 't b (l1 l2) (p1 p2 d) -> t b d (l1 p1) (l2 p2)', l1=l1, l2=l2, p1=self.patch_dim, p2=self.patch_dim)
        return y

    def reset(self, mask):
        self.transformer.reset(mask)


