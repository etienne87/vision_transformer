import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as f
from core.transformer import FeedForward


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * ((d // h) ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class QuerySetAttention(nn.Module):
    def __init__(self, num_queries, dim, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        nn.init.xavier_uniform_(self.queries)
        self.attn_in = SelfAttention(dim, heads)
        
        #fixme: to remove
        self.attn_out = SelfAttention(dim, heads)

    def forward(self, x):
        b = x.shape[0]
        q = self.queries.expand(b, -1, -1)
        q_out = self.attn_in(q, x)
        return q_out


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        #self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        #self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.slots = nn.Parameter(torch.randn(1, self.num_slots, dim))  
        nn.init.xavier_uniform_(self.slots)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        slots = self.slots.expand(b, -1, -1)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


if __name__ == '__main__':
    b, n, d = 5, 32, 16
    x = torch.randn(b,n,d)
    qsa = QuerySetAttention(3, 16, 8)
    sla = SlotAttention(4, 16)  
    y = qsa(x)
    y2 = sla(x)
    print(y2.shape)
    print(y.shape)

