import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as f
from core.relative_embedding import RelativePositionBias


def normalize(t, eps=1e-8):
    """
    Normalized Attention Without Probability Cage
    Oliver Richter, Roger Wattenhofer

    https://arxiv.org/abs/2005.09561
    """
    t -= t.mean(dim=-1, keepdim=True)
    s = (t ** 2).mean(dim=-1, keepdim=True)
    return t * torch.rsqrt(s + eps)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        return self.norm(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.scale = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreBatchNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = rearrange(x, 'b l d -> b d l')
        x = self.norm(x)
        x = rearrange(x, 'b d l -> b l d')
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., dim_qk=None, normalize_fn=lambda x:torch.nn.functional.softmax(x, dim=-1)):
        super().__init__()
        self.heads = heads
        dim_qk = dim_qk if dim_qk is not None else dim

        self.to_qk = nn.Linear(dim, dim_qk * 2, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.normalize_fn = normalize_fn

        #learning scale is not so useful but perhaps it turns softmax into max or average?
        self.scale = dim ** -0.5
        # self.scale = nn.Parameter(torch.randn((1,heads,1,1), dtype=torch.float32))
        # nn.init.normal_(self.scale, mean=dim**-0.5, std=0.01)

    def forward(self, x, mask = None, pos = None):
        b, n, _, h = *x.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        qkv = *self.to_qk(x).chunk(2, dim = -1), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # position "infused" attention: inject position embedding to q & k
        if pos is not None:
            # pos embedding is shared everywhere... might be a bit too biased...
            hpos = rearrange(pos, 'b n (h d) -> b h n d', h = h)
            q += hpos
            k += hpos

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = f.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # attn = dots.softmax(dim=-1)
        attn = self.normalize_fn(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class TransformerAN(nn.Module):
    """all normalization"""
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PostNorm(dim, Attention(dim, heads = heads, dropout = dropout, normalize_fn=normalize))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, rezero=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        #ReZero
        for _ in range(depth):
            if rezero:
                self.layers.append(nn.ModuleList([
                        ReZero(Attention(dim, heads=heads, dim_qk=None, dropout=dropout)),
                        ReZero(FeedForward(dim, mlp_dim, dropout=dropout))
                    ]))
            else:
                for _ in range(depth):
                    self.layers.append(nn.ModuleList([
                        Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
                    ]))

    def forward(self, x, mask = None, pos=None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, pos=pos)
            x = ff(x)
        return x




if __name__ == '__main__':

    b,n,c = 3,50,32
    x = torch.randn(b,n,c)
    net = Transformer(dim=c, depth=3, heads=8, mlp_dim=64, dropout=0.)
    y = net(x)
    print(y.shape)
