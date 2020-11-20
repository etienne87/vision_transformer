import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from core.transformer import Attention, PreNorm, Residual, FeedForward



class RecursiveTransformer(nn.Module):
    """Simplest possible Recursive Transformer
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.attn = Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)))
        self.ff = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
        self.depth = depth

    def forward(self, x, mask = None):
        for i in range(self.depth): 
            x = self.attn(x, mask = mask)
            x = self.ff(x)
        return x


# Clusformer = pool and then unpool with skip connection
# class Clusformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, dropout):
#         super().__init__()


# Hopfield = reuse attention map with current K to form new Q
class HopfieldAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, prev_attn = None, mask = None):
        b, n, _, h = *x.shape, self.heads
        kv = self.to_k(x), self.to_v(x)
        reshape = lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h)
        k, v = map(reshape, kv)

        if prev_attn is None:
            q = self.to_q(x)
            q = reshape(q)
        else:
            q = torch.einsum('bhij,bhjd->bhid', prev_attn, k)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out, attn


class RecursiveHopfield(nn.Module):
    """Recursive Hopfield 
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.attn = PreNorm(dim, HopfieldAttention(dim, heads = heads, dropout = dropout))
        self.ff = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
        self.depth = depth

    def forward(self, x, mask = None):
        prev_attn = None
        for i in range(self.depth): 
            r, prev_attn = self.attn(x, prev_attn = prev_attn, mask = mask)
            x = x + r
            x = self.ff(x)
        return x
