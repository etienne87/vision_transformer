"""
Code Taken from: https://github.com/lucidrains/axial-attention/blob/master/axial_attention/axial_attention.py
"""
import torch
import torch.nn as nn
from operator import itemgetter
from core.transformer import * 


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


# also calculates the inverse permutation to bring the tensor back to its original shape
def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


def calculate_permutation(num_dimensions, axial_dim, emb_dim):
    """calculate one permutation to bring input tensor attendable
    to specified axial dimension
    """
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)

    last_two_dims = [axial_dim, emb_dim]
    dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
    permutation = [*dims_rest, *last_two_dims]
    return permutation


def calculate_permutations(num_dimensions, emb_dim):
    """calculates all permutations to bring input tensor to something attend-able
    """
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
      
    return permutations


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_index = -1, sum_axial_out = True, dropout = 0.):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []

        test = calculate_permutations(num_dimensions, dim_index)

        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, Attention(dim, heads, dropout)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert x.dim() == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        axial_attn = self.axial_attentions[0]
        out = axial_attn(x)
        for axial_attn in self.axial_attentions[1:]:
            out = axial_attn(out)
        return out



class AxialTransformer(nn.Module):
    """Axial Transformer
    Will work in whatever number of topological dimensions
    TODO: add mask
    """
    def __init__(self, num_dimensions, dim, depth, heads, mlp_dim, dropout, rezero=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        for _ in range(depth):
            if rezero:
                self.layers.append(nn.ModuleList([
                    ReZero(AxialAttention(dim, num_dimensions, heads = heads, dim_index=-1, dropout = dropout)),
                    ReZero(FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, AxialAttention(dim, num_dimensions, heads = heads, dim_index=-1, dropout = dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
                ]))

    def forward(self, x):
        assert x.shape[-1] == self.dim
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


