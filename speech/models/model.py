from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_dim, config):
        """
        Parameters
        -----------
        conv_cfg: [out_c]
            list of list that define the parameters of the convolutions with format [out_c, h, w, s] 
            where out_c is the size of the output channel, h and w specify the height and width of the kernel or filter, 
            and s is the stride of the kernel and the stride is symmetric
        """
        
        super().__init__()
        self.input_dim = input_dim

        encoder_cfg = config["encoder"]
        conv_cfg = encoder_cfg["conv"]

        convs = []
        in_c = 1
        for out_c, h, w, s_h, s_w, p_h, p_w in conv_cfg:     
            conv = nn.Conv2d(in_channels=in_c, 
                             out_channels=out_c, 
                             kernel_size=(h, w),
                             stride=(s_h, s_w), 
                             padding=(p_h, p_w))
            batch_norm =  nn.BatchNorm2d(out_c)
            convs.extend([conv, batch_norm, nn.ReLU()])
            #convs.extend([conv, nn.ReLU()])
            if config["dropout"] != 0:
                convs.append(nn.Dropout(p=config["dropout"]))
            in_c = out_c

        self.conv = nn.Sequential(*convs)
        conv_out = out_c * self.conv_out_size(self.input_dim, 1)

        assert conv_out > 0, \
          "Convolutional ouptut frequency dimension is negative."

        rnn_cfg = encoder_cfg["rnn"]
        assert rnn_cfg["type"] == "GRU" or rnn_cfg["type"] == "LSTM", "RNN type in config not supported"

        self.rnn = eval("nn."+rnn_cfg["type"])(
                        input_size=conv_out,
                        hidden_size=rnn_cfg["dim"],
                        num_layers=rnn_cfg["layers"],
                        batch_first=True, dropout=config["dropout"],
                        bidirectional=rnn_cfg["bidirectional"])
        self._encoder_dim = rnn_cfg["dim"]

    def conv_out_size(self, n, dim):
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                # assuming a valid convolution meaning no padding
                k = c.kernel_size[dim]
                s = c.stride[dim]
                p = c.padding[dim]
                n = (n - k + 1 + 2*p) / s
                n = int(math.ceil(n))
        return n

    def forward(self, batch):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def encode(self, x, rnn_args=None):
        """this function processes the input data x through the CNN and RNN layers specified
            in the model encoder config.

        """
        x = x.unsqueeze(1)      
        
        x = self.conv(x)
        # At this point x should have shape
        # (batch, channels, time, freq)
        
        x = torch.transpose(x, 1, 2).contiguous()
        
        # Reshape x to be (batch, time, freq * channels)
        # for the RNN
        
        b, t, f, c = x.data.size()
        x = x.view((b, t, f*c)) 
        #x = x.view((x.data.size()[0], x.data.size()[1], -1)) 

        x, rnn_args = self.rnn(x, rnn_args)
        
        # if self.rnn.bidirectional:
        #     half = x.size()[-1] // 2
        #     x = x[:, :, :half] + x[:, :, half:]

        return x, rnn_args

    def loss(self, x, y):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def set_eval(self):
        """
        Set the model to evaluation mode.
        """
        self.eval()

    def set_train(self):
        """
        Set the model to training mode.
        """
        self.train()

    def infer(self, x):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def is_cuda(self):
        return list(self.parameters())[0].is_cuda

    @property
    def encoder_dim(self):
        return self._encoder_dim

class LinearND(nn.Module):

    def __init__(self, *args):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*args)

    def forward(self, x):
        size = x.size()
        n = int(np.prod(size[:-1]))
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

def zero_pad_concat(inputs):
    """this loops over all of the examples in inputs and adds them 
    to the zero's array input_mat so that for examples with length less 
    than the max have zero's from the end of the example until max_t
    """
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)

    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0], :] = inp
    return input_mat

