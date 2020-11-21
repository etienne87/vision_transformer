import torch

from core import *
from arch import * 


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


def hopfield():
    b,n,c = 3,50,32
    x = torch.rand(b,n,c)
    net = HopfieldAttention(dim=c, heads=8)
    y, att = net(x)
    z, _ = net(y, att)
    print(z.shape)


def recursive_hopfield():
    b,n,c = 3,50,32
    x = torch.rand(b,n,c)
    net = RecursiveHopfield(dim=c, depth=3, heads=8, mlp_dim=64, dropout=0.)
    y = net(x)
    print(y.shape)


def vit():
    b,c,h,w = 10,3,64,64
    x = torch.randn(b,c,h,w)
    net = ViT(c,6)
    y = net(x)
    print(y.shape)


def vit3d():
    t,b,c,h,w = 16,10,3,64,64
    x = torch.randn(t,b,c,h,w)
    net = ViT3d(c,6)
    y = net(x)
    print(y.shape)


def att_rnn():
    t,b,c,h,w = 16,10,3,16,16
    x = torch.randn(t,b,c,h,w)
    net = ImagesAttentionRNN(c, 16, 8)
    y = net(x)
    print(y.shape)


def axial_attention():
    b,c,h,w,z = 5,8,28,29,30 
    x = torch.randn(b,c,h,w,z)
    net = AxialAttention(c, num_dimensions=3, dim_index=1)
    y = net(x) 
    print(y.shape)


def axial_positional_embedding():
    b,c,h,w,z = 5,8,28,29,30 
    x = torch.randn(b,c,h,w,z)
    net = AxialPositionalEmbeddingVolume(c, (28,29,30))
    y = net(x) 
    print(y.shape)


def axial_transformer():
    b,c,h,w,z = 5,8,28,29,30 
    x = torch.randn(b,h,w,z,c)
    net = AxialTransformer(3, c, 1, 2, 64, 0)
    y = net(x)
    print(y.shape)


def axial_vit3d():
    t,b,c,h,w = 12,4,3,64,64
    x = torch.randn(t,b,c,h,w)
    net = AxialViT3d(3, 11)
    y = net(x)
    print(y.shape)


def unet_rnn():
    t,b,c,h,w = 17,3,16,32,32
    x = torch.randn(t,b,c,h,w)
    net = UnetRNN(c, 32)
    net.reset()
    y = net(x)
    print(y.shape)

if __name__ == '__main__':
    import fire
    fire.Fire()
