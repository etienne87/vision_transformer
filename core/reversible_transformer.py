import math
import torch
import torch.nn as nn
import revtorch as rv
from core.transformer import PreNorm, Attention, FeedForward


class RevZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.scale = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale 

class ReversibleTransformer(nn.Module):
    """O(1) backprop memory cost thanks revtorch"""
    def __init__(self, dim, depth, heads, mlp_dim, dropout, rezero=False):
        super().__init__()

        self.entry = nn.Parameter(torch.FloatTensor([0]))
        blocks = []
        for _ in range(depth):
            if rezero:
                f_func = RevZero(Attention(dim, heads=heads, dropout=dropout))
                g_func = RevZero(FeedForward(dim, mlp_dim, dropout=dropout))
            else:
                f_func = PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))
                g_func = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            block = rv.ReversibleBlock(f_func, g_func)
            blocks.append(block)

        self.layers = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x):
        x = self.entry + x
        x = self.layers(x)
        return x



if __name__ == '__main__':
    b,n,c = 3,50,32
    x = torch.randn(b,n,c)
    net = ReversibleTransformer(dim=c, depth=174, heads=8, mlp_dim=64, dropout=0.) 
    #y = test(x) 
    y = net(x)
    y[0,0,0].backward()

