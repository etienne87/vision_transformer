import math
import torch
import torch.nn as nn
from core.transformer import Attention, Residual, PreNorm 


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-math.sqrt(6 / self.in_features) / self.omega_0, 
                                             math.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, x):
        y = torch.sin(self.omega_0 * self.linear(x))
        return y


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-math.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              math.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output 


class SirenPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.net = Siren(2, hidden_dim, 2, embedding_dim, True)
        self.positions = None

    def forward(self, x, width):
        if self.positions is None or self.positions.size(1) != x.size(1):
            seq_length = x.size(1)
            positions = torch.arange(seq_length, dtype=torch.int32, device=x.device)
        else:
            positions = self.positions

        cx = positions%width
        cy = positions//width
        positions = torch.cat((cx[:,None],cy[:,None]), dim=1).float()/width

        position_embeddings = self.net(positions)[None]

        return x + position_embeddings


class SineFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = Siren(dim, hidden_dim, 1, dim) 

    def forward(self, x):
        return self.net(x)


class SineTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                PreNorm(dim, SineFeedForward(dim, mlp_dim))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

