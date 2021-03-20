
"""
Transformer for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
import revtorch as rv

from functools import wraps
from einops import rearrange, repeat


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


def fourier_encode(x, num_encodings = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        n = torch.norm(x, dim = -1, keepdim = True).clamp(min = self.eps)
        return x / n * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = ScaleNorm(dim)
        self.norm_context = ScaleNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * f.gelu(gates)


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

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)



class SparseInputPerceiver(nn.Module):
    """Sparsify RGBT volume
    -> Perceiver Strategy (Cross-Attention + iterative refinements)
    """
    def __init__(self,
                input_channels,
                output_channels,
                depth=5,
                input_axis=3,
                num_fourier_features=64,
                num_latents=128,
                latent_dim=512,
                latent_heads=8,
                cross_dim=512,
                cross_heads=1,
                weight_tie_layers=False):
        super().__init__()
        self.num_fourier_features = num_fourier_features
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.cross_dim = cross_dim
        self.latent_heads = latent_heads
        self.input_dim = 4**2 * input_channels + input_axis * ((num_fourier_features * 2) + 1)

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.pos_emb = nn.Parameter(torch.randn(num_latents, latent_dim))

        # get_cross_attn = lambda: ReZero(Attention(latent_dim, self.input_dim))
        # get_cross_ff = lambda: ReZero(FeedForward(latent_dim))
        # get_latent_attn = lambda: ReZero(Attention(latent_dim))
        # get_latent_ff = lambda: ReZero(FeedForward(latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, self.input_dim), context_dim = self.input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))

        if weight_tie_layers:
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_cross_attn(),
                get_cross_ff(),
                get_latent_attn(),
                get_latent_ff()
            ]))

        self.out_cross_attn = Attention(self.input_dim, latent_dim)

        # we will predict trajectories as
        # x_i,t = predict_a(latent_i) * t + predict_b(latent_i)
        # self.to_boxes_a = nn.Linear(latent_dim, 4)
        # self.to_boxes_b = nn.Linear(latent_dim, 4)
        # self.to_logits = nn.Linear(latent_dim, output_channels-4)

        self.to_logits = nn.Linear(latent_dim, output_channels)

    def sparsify_v1(self, input, threshold=0.3):
        """
        #A. Sparsify x into:
            #1. positions   x,y,t   B,N,3
            #2. values      r,g,b   B,N,3
            #3. masks               B,N,
        """
        b,h,w,nt,c = input.shape
        mask = input.max(dim=-1)[0].abs() > threshold
        nums = mask.view(b,-1).sum(dim=-1)
        max_len = nums.max().item()
        dtype = input.dtype
        device = input.device

        pos = torch.zeros((b,max_len,3), dtype=dtype, device=device)
        vals = torch.zeros((b,max_len,c), dtype=dtype, device=device)
        masks = torch.zeros((b,max_len,), dtype=torch.bool, device=device)

        for i in range(b):
            y, x, t = torch.where(mask[i])
            n = len(y)
            pos[i, :n, 0] = 2*y/h-1
            pos[i, :n, 1] = 2*x/w-1
            pos[i, :n, 2] = 2*t/nt-1
            masks[i, :n] = 1
            vals[i, :n] = input[i, y, x, t]
        return pos, vals, masks

    #def sparsify_v2(self, input, threshold=0):
    #    """sends all rgb values where rgb_t+1 != rgb_t"""

    def perceiver(self, x, data, mask):
        """
        #B. Run Perceiver(positions, values, masks)
        """
        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x
            x = latent_attn(x) + x
            x = latent_ff(x) + x
        return x

    def forward(self, input):
        """
        1. we expect x: B,T,H,W,C
        2. sparsify with thresholding or with another technique
            (e.g: what about outputting only where difference is greater than threshold?)
        3. so far: predict from latent space only
        """
        num_tbins = len(input)
        with torch.no_grad():
            input = rearrange(input, 't b c h w -> b h w t c')
            #patchify
            p = 4
            input = rearrange(input, 'b (h p1) (w p2) t c -> b h w t (p1 p2 c)', p1 = p, p2 = p)
            b = len(input)
            pos, vals, masks = self.sparsify_v1(input)
            enc_pos = fourier_encode(pos, self.num_fourier_features)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')

        data = torch.cat((vals, enc_pos), dim = -1)
        data = rearrange(data, 'b ... d -> b (...) d')

        latents = self.latents + self.pos_emb # why not just one set of params
        latents = repeat(latents, 'n d -> b n d', b = b)
        latents = self.perceiver(latents, data, masks)

        # linear model of trajectories prediction
        # in theory we should use linear assignement at the trajectory level...
        # time = torch.linspace(0,1,num_tbins)[:,None,None,None].to(latents)
        # c1 = self.to_boxes_a(latents)[None,:]
        # c2 = self.to_boxes_b(latents)[None,:]
        # boxes = c1 * time + c2
        # logits = self.to_logits(latents)
        # logits = repeat(logits, 'b n d -> t b n d', t = num_tbins)
        # output = torch.cat((boxes, logits), dim=-1)

        output = self.to_logits(latents)
        output = rearrange(output, 'b (n t) d -> t b n d', t = num_tbins)
        output = output.contiguous()
        return output




if __name__ == '__main__':
    b,c,h,w,t = 3,3,64,64,10
    x = torch.randn(t,b,c,h,w)
    x = (x.abs() > 2) * x
    net = SparseInputPerceiver(3, 11+4)
    y = net(x)
    print(y.shape)




