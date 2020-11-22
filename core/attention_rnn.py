import math
import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange

from core.transformer import Attention, Transformer
from core.positional_encoding import LearnedPositionalEncoding 
from core.temporal import seq_wise





class AttentionRNN(nn.Module):
    """Attention applied multiple timesteps
    Concatenates memory to input and applies self-attention.
    """

    def __init__(self, in_channels, embedding_dim, memory_tokens, num_layers=1, num_heads=8, hidden_dim=32, max_pos_embeddings=512, dropout=0.):
        super().__init__()
        self.in_channels = in_channels 
        self.memory_tokens = memory_tokens
        self.embedding_dim = embedding_dim
        self.prev_h = torch.zeros((1, memory_tokens, embedding_dim), dtype=torch.float32)
        self.xh = nn.Linear(in_channels, embedding_dim)

        self.pex = LearnedPositionalEncoding(max_pos_embeddings, embedding_dim)
        self.transformer = Transformer(embedding_dim, num_layers, num_heads, hidden_dim, dropout) 

    def forward(self, x):
        xseq = x.unbind(0)
        if len(self.prev_h) != x.size(1):
            device = x.device
            self.prev_h = torch.zeros((x.size(1), self.memory_tokens, self.embedding_dim), dtype=torch.float32).to(device)

        self.prev_h.detach_()

        result = []
        for t, xt in enumerate(xseq):
            #1. cat prev_h,x_flat
            xh = self.xh(xt)

            cat = torch.cat((self.prev_h, xh), dim=1)
            cat = self.pex(cat)
            
            #2. apply self-attention
            y = self.transformer(cat)

            #3. split h, x_out
            h, out = torch.split(y, [self.memory_tokens, y.size(1)-self.memory_tokens], dim=1)
            
            #4. reshape back!
            result.append(out.unsqueeze(0))

            self.prev_h = h

        res = torch.cat(result, dim=0)
        return res

    @torch.jit.export
    def reset(self, mask):
        """To be called in between batches"""
        if len(self.prev_h) == len(mask):
            self.prev_h.detach_()
            self.prev_h *= mask[...,0]



""" For interfacing with a normal conv architecture
"""
def flatten(x):
    x = rearrange(x, 'b c ... -> b (...) c')
    return x

def unflatten(x, *dims):
    pat = ""
    variables = {}
    for i, dim in enumerate(dims):
        key = chr(ord('h')+i)
        pat += key+ ' '
        variables[key] = dim
    sent = 'b ('+pat+') c -> b c ' + pat 
    x = rearrange(x, sent, **variables)
    return x


class ImagesAttentionRNN(AttentionRNN):
    def __init__(self, in_channels, embedding_dim, memory_tokens, num_layers=1, num_heads=8, hidden_dim=32, max_pos_embeddings=512):
        super().__init__(in_channels, embedding_dim, memory_tokens, num_layers, num_heads, hidden_dim, max_pos_embeddings)

    def forward(self, x):
        h,w = x.shape[-2:]
        x = seq_wise(flatten)(x)
        x = super().forward(x)
        x = seq_wise(unflatten)(x, h, w)
        return x

class VolumesAttentionRNN(AttentionRNN):
    def __init__(self, in_channels, embedding_dim, memory_tokens, num_layers=1, num_heads=8, hidden_dim=32, max_pos_embeddings=512):
        super().__init__(in_channels, embedding_dim, memory_tokens, num_layers, num_heads, hidden_dim, max_pos_embeddings)

    def forward(self, x):
        h,w,z = x.shape[-3:]
        x = seq_wise(flatten)(x)
        x = super().forward(x)
        x = seq_wise(unflatten)(x, h,w,z)
        return x
