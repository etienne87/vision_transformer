
"""
Transformer for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as f


from core.transformer import Attention


def fourier_encode(x, num_encodings = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.scale = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 75, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)



class SparsePerceiver(nn.Module):
    """Sparsify RGBT volume
    -> Perceiver Strategy (Cross-Attention + iterative refinements)
    """
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()

        self.patch_dim = patch_dim

        self.flatten_dim_in = patch_dim * patch_dim * in_channels
        self.linear_encoding = nn.Linear(self.flatten_dim_in, embedding_dim)

        self.position_encoding = LearnedPositionalEncoding(max_len, embedding_dim)

        self.transformer = Transformer(embedding_dim, num_layers, num_heads, hidden_dim, dropout)

        self.flatten_dim_out = patch_dim * patch_dim * out_channels
        self.linear_decoding = nn.Linear(embedding_dim, self.flatten_dim_out)


    def forward(self, x):
        t,b,c,h,w = x.shape
        #A. Sparsify x into:
            #1. positions   x,y,t   B,N,3
            #2. values      r,g,b   B,N,3
            #3. masks               B,N,

        #B. Run Perceiver(positions, values, masks)

        #C. Cross-Attention to input positions
            #1. positions x,y,t
            #2. output values B,out_channels






