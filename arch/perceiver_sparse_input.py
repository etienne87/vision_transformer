
"""
Transformer for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

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
                depth=3,
                input_axis=3,
                num_fourier_features=64,
                num_latents=32,
                latent_dim=512,
                latent_dim_head = 64,
                latent_heads=8,
                cross_dim=512,
                cross_heads=1,
                cross_dim_head=64,
                weight_tie_layers=False):
        super().__init__()
        self.num_fourier_features = num_fourier_features
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.cross_dim = cross_dim
        self.latent_heads = latent_heads
        self.cross_dim_head = cross_dim_head
        self.latent_dim_head = latent_dim_head
        self.input_dim = input_channels + input_axis * ((num_fourier_features * 2) + 1)

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.pos_emb = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: ReZero(Attention(latent_dim, self.input_dim))
        get_cross_ff = lambda: ReZero(FeedForward(latent_dim))
        get_latent_attn = lambda: ReZero(Attention(latent_dim))
        get_latent_ff = lambda: ReZero(FeedForward(latent_dim))

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
        self.to_logits = nn.Linear(self.latent_dim, output_channels)

    def sparsify_v1(self, input, threshold=0):
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
            pos[i, :n, 0] = y
            pos[i, :n, 1] = x
            pos[i, :n, 2] = t
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

    # def predict(self, data, latents):
    #     """
    #     #C. Cross-Attention to input positions not all of them
    #         #1. positions x,y,t
    #         #2. output values B,out_channels
    #     """
    #     x = self.out_cross_attn(data, latents) + data
    #     y = self.to_logits(x)
    #     return y

    # def densify(self, positions, predictions, input):
    #     b,h,w,t,_ = input.shape
    #     output = torch.zeros((b,h,w,t,self.output_channels), dtype=input.dtype, device=output.device)

    def forward(self, input):
        """
        1. we expect x: B,T,H,W,C
        2. sparsify with thresholding or with another technique
            (e.g: what about outputting only where difference is greater than threshold?)
        3. so far: predict from latent space only
        """
        input = rearrange(input, 't b c h w -> b h w t c')
        b = len(input)

        pos, vals, masks = self.sparsify_v1(input)

        enc_pos = fourier_encode(pos, self.num_fourier_features)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')

        data = torch.cat((vals, enc_pos), dim = -1)
        data = rearrange(data, 'b ... d -> b (...) d')

        latents = self.latents + self.pos_emb # why not just one set of params
        latents = repeat(latents, 'n d -> b n d', b = b)
        latents = self.perceiver(latents, data, masks)

        return self.to_logits(latents)





if __name__ == '__main__':
    b,c,h,w,t = 3,3,64,64,10
    x = torch.randn(b,h,w,t,c)
    x = (x.abs() > 2) * x
    net = SparseInputPerceiver(3, 10)
    y = net(x)
    print(y.shape)




