import torch

from core.transformer import *
from arch.vit import * 


def learned_positional_encoding():
    b,n,c = 3,50,32
    x = torch.randn(b,n,c)
    pe = LearnedPositionalEncoding(100, c)
    y = pe(x)
    print(y.shape)


def transformer():
    b,n,c = 3,50,32
    x = torch.randn(b,n,c)
    net = Transformer(dim=c, depth=3, heads=8, mlp_dim=64, dropout=0.) 
    y = net(x)
    print(y.shape)


def vit():
    b,c,h,w = 10,3,64,64
    x = torch.randn(b,c,h,w)
    net = ViT(c,6)
    y = net(x)
    print(y.shape)



if __name__ == '__main__':
    import fire
    fire.Fire()
