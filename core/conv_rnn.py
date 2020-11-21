import torch
import torch.nn as nn
import torch.nn.init as init

from core.conv import ConvLayer
from core.temporal import SequenceWise


class RNNCell(nn.Module):
    """
    Abstract class that has memory. serving as a base class to memory layers.

    Args:
        hard (bool): Applies hard gates to memory updates function.
    """

    def __init__(self, hard):
        super(RNNCell, self).__init__()
        self.set_gates(hard)

    def set_gates(self, hard):
        if hard:
            self.sigmoid = self.hard_sigmoid
            self.tanh = self.hard_tanh
        else:
            self.sigmoid = torch.sigmoid
            self.tanh = torch.tanh

    def hard_sigmoid(self, x_in):
        x = x_in * 0.5 + 0.5
        y = torch.clamp(x, 0.0, 1.0)
        return y

    def hard_tanh(self, x):
        y = torch.clamp(x, -1.0, 1.0)
        return y

    def reset(self):
        raise NotImplementedError()


class ConvLSTMCell(RNNCell):
    """ConvLSTMCell module, applies sequential part of LSTM.

    LSTM with matrix multiplication replaced by convolution
    See Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
    (Shi et al.)

    Args:
        hidden_dim (int): number of output_channels of hidden state.
        kernel_size (int): internal convolution receptive field.
        conv_func (fun): functional that you can replace if you want to interact with your 2D state differently.
        hard (bool): applies hard gates.
        dropout_p (float): dropout probability.
    """

    def __init__(self, hidden_dim, kernel_size, conv_func=nn.Conv2d, hard=False, dropout_p=0):
        super(ConvLSTMCell, self).__init__(hard)
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=True)

        self.prev_h = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)
        self.prev_c = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)
        self.reset_parameters()

        self.forget_bias = 1.0

        self.dropout_p = dropout_p
        self.dropout: Optional[nn.Module] = None
        self.dropout_p = dropout_p
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(self.dropout_p)

    def reset_parameters(self):
        init.xavier_uniform_(self.conv_h2h.weight)
        init.constant_(self.conv_h2h.bias, val=0.0)
        hidden_dim = self.hidden_dim
        init.orthogonal_(self.conv_h2h.weight[:hidden_dim, ...])
        init.orthogonal_(self.conv_h2h.weight[hidden_dim:2 * hidden_dim, ...])
        init.orthogonal_(self.conv_h2h.weight[2 * hidden_dim:3 * hidden_dim, ...])
        init.orthogonal_(self.conv_h2h.weight[3 * hidden_dim:, ...])

    @torch.jit.export
    def get_dims_NCHW(self):
        return self.prev_h.size()

    def forward(self, x):
        assert x.dim() == 5

        xseq = x.unbind(0)

        assert self.prev_h.size() == self.prev_c.size()

        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_h.size()
        input_N, input_C, input_H, input_W = xseq[0].size()
        assert input_C == 4 * hidden_C
        assert hidden_C == self.hidden_dim

        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W or self.prev_h.device != x.device:
            device = x.device
            self.prev_h = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)
            self.prev_c = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)

        self.prev_h.detach_()
        self.prev_c.detach_()

        result = []
        for t, xt in enumerate(xseq):
            assert xt.dim() == 4

            tmp = self.conv_h2h(self.prev_h) + xt

            cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
            i = self.sigmoid(cc_i)
            f = self.sigmoid(cc_f + self.forget_bias)
            o = self.sigmoid(cc_o)
            g = self.tanh(cc_g)

            if self.dropout_p > 0:
                g = self.dropout(g)

            c = f * self.prev_c + i * g
            h = o * self.tanh(c)
            result.append(h.unsqueeze(0))

            self.prev_h = h
            self.prev_c = c

        res = torch.cat(result, dim=0)
        return res

    @torch.jit.export
    def reset(self, mask):
        """Sets the memory (or hidden state to zero), normally at the beginiing of a new sequence.

        `reset()` needs to be called at the beginning of a new sequence. The mask is here to indicate which elements
        of the batch are indeed new sequences. """
        batch_size, _, _, _ = self.prev_h.size()
        if batch_size == len(mask) and mask.device == self.prev_h.device:
            assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1]) or mask.shape == torch.Size([1])
            self.prev_h.detach_()
            self.prev_c.detach_()
            self.prev_h *= mask
            self.prev_c *= mask

    @torch.jit.export
    def reset_all(self):
        """Resets memory for all sequences in one batch."""
        self.reset(torch.zeros((len(self.prev_h), 1, 1, 1), dtype=torch.float32, device=self.prev_h.device))


class ConvRNN(nn.Module):
    """ConvRNN module. ConvLSTM cell followed by a feed forward convolution layer.

    Args:
        in_channels (int): number of input channels
        out_chanels (int): number of output channels
        kernel_size (int): separable conv receptive field
        stride (int): separable conv stride.
        padding (int): padding.
        separable (boolean): if True, uses depthwise seprable convolution for the forward convolutional layer.
        **kwargs: additional parameters for the feed forward convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 cell='lstm', separable=False, **kwargs):
        assert cell == 'lstm', "Only 'lstm' cells are supported for the time being"
        super(ConvRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_x2h = SequenceWise(ConvLayer(in_channels, 4 * out_channels, activation='Identity', stride=stride,
                                               separable=separable, **kwargs))
        self.timepool = ConvLSTMCell(out_channels, 3)

    def forward(self, x):
        y = self.conv_x2h(x)
        h = self.timepool(y)
        return h

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        """Resets memory of the network."""
        self.timepool.reset(mask)

    @torch.jit.export
    def reset_all(self):
        self.timepool.reset_all()
