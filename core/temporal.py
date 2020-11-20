import torch
import torch.nn as nn


def time_to_batch(x):
    """flatten batch & time together """
    t, n = x.size()[:2]
    x = x.view(n * t, *x.size()[2:])
    return x, n


def batch_to_time(x, n=32):
    """unflatten batch & time"""
    nt = x.size(0)
    time = nt // n
    x = x.view(time, n, *x.size()[1:])
    return x


def seq_wise(function):
    """Decorator to apply 4 dimensionnal tensor functions on 5 dimensional temporal tensor input."""
    def sequence_function(x5, *args, **kwargs):
        x4, batch_size = time_to_batch(x5)
        y4 = function(x4, *args, **kwargs)
        return batch_to_time(y4, batch_size)
    return sequence_function


class SequenceWise(nn.Sequential):
    """Class Wrapper"""
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        if x.dim() == 4:
            return super().forward(x)
        else:
            return seq_wise(super().forward)(x)
