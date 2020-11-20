import math
import torch
import torch.nn as nn



class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super().__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim):
        super().__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.positions = None

    def forward(self, x):
        if self.positions is None:
            seq_length = x.size(1)
            positions = torch.arange(seq_length, device=x.device).expand((1, -1)) 
        else:
            positions = self.positions

        position_embeddings = self.pe(positions).repeat(x.size(0),1,1)
        return x + position_embeddings
