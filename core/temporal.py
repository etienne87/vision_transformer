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


class StateFulCell(nn.Module):
    """
    StateFulCell, applies a recurrent cell to a feedforward layer
    and keeps track of the hidden state.

    Args:
        layer: feed-forward layer
        init_hidden: initialize hidden from current input
        cat: how to cat x & hidden
        split: how to split x from hidden
    """
    def __init__(self, layer, init_hidden, cat, split):
        super().__init__(self)
        self.layer = layer_fn
        self.init_hidden = init_hidden
        self.cat = cat

    def forward(self, x):
        if self.hidden is None:
            self.hidden = self.init_hidden(x)

        x = self.cat(x, self.hidden)
        y = self.layer(x)
        y, self.hidden = self.split(y)
        return y

    def reset(self, mask):
        """
        Args:
            mask: shape (B,)
        """
        self.hidden.detach()
        ndim = self.hidden.ndim-1
        mask = mask.reshape(mask.shape + (1,)*ndim)
        self.hidden *= mask


class StateFulRnn(nn.Module):
    """
    StateFulRnn, owns StateFulCell and applies it to a sequence.
    """
    def __init__(self, layer, init_hidden, cat, split):
        super().__init__(self)
        self.cell = StateFulCell(layer, init_hidden, cat, split)

    def forward(self, x, memory_mask):
        """
        Args:
            x: shape (T,B,*)
            memory_mask: shape (B,)
        """
        self.cell.reset(memory_mask)
        x = x.unbind(0)
        out = []
        for x_t in x:
            y_t = self.cell(x_t)
            out.append(y_t[None])

        out = torch.cat(out)
        return out








