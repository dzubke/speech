from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from speech.models import ctc_model
from speech.models import model
from .ctc_decoder import decode
from .ctc_decoder_dist import decode_dist



class CTC_train(ctc_model.CTC):
    def __init__(self, freq_dim, output_dim, config):
        super().__init__(freq_dim, output_dim, config)

        # include the blank token
        self.blank = output_dim
        self.fc = model.LinearND(self.encoder_dim, output_dim + 1)

    def forward(self, x, rnn_args=None, softmax=False):
        # softmax should be true for inference, false for loss calculation
        #x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_impl(x, rnn_args,  softmax=softmax)

    def forward_impl(self, x, rnn_args=None, softmax=False):
        if self.is_cuda:
            x = x.cuda()

        # padding is half the filters of the 3 conv layers. 
        # conv.children are: [Conv2d, BatchNorm2d, ReLU, Dropout, Conv2d, 
        # BatchNorm2d, ReLU, Dropout, Conv2d, BatchNorm2d, ReLU, Dropout]
        # conv indicies with batch norm: 0, 4, 8
        # conv layer indicies without batch norm: 0, 3, 6
        pad = list(self.conv.children())[0].kernel_size[0]//2 + \
            list(self.conv.children())[4].kernel_size[0]//2 + \
            list(self.conv.children())[8].kernel_size[0]//2
        x = nn.functional.pad(x, (0,0,pad,pad))

        x, rnn_args = self.encode(x, rnn_args)    
        x = self.fc(x)          
        if softmax:
            return torch.nn.functional.softmax(x, dim=2), rnn_args
        return x, rnn_args

    def native_loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        print(f"input size: {x.size()}, labels size: {y.size()}")
        print(f"labels preview: {y[:278]}")
        print(f"input lens size: {x_lens.size()}, label lens size: {y_lens.size()}")
        print(f"input lens: {x_lens}, label lens: {y_lens}")
               
 
        out, rnn_args = self.forward_impl(x, softmax=False)
        log_probs = nn.functional.log_softmax(out, dim=2)

        # reshaep the log_probs and labels y
        print(f"original log_probs size: {log_probs.size()}")
        print(f"permuted size: {log_probs.permute(1,0,2).size()}")
        print(f"labels rehape: {y.reshape(8, -1).size()}")
        print(f"labels reshape preview: {y.reshape(8, -1)[:2]}")

        #blank_idx = out.size()[-1] - 1
        loss_fn = torch.nn.CTCLoss(blank=self.blank, reduction='none')        # native ctc loss     
        loss = loss_fn(log_probs.permute(1,0,2), y.reshape(8, -1), x_lens, y_lens)
        print(f"output size: {out.size()}, loss size: {loss.size()}")
        print(f"loss value: {loss}")
        
        return loss

    def collate(self, inputs, labels):
        max_t = max(i.shape[0] for i in inputs)
        max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        x = torch.FloatTensor(model.zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]

        return batch
    
    def infer(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        probs, rnn_args = self.forward_impl(x, softmax=True)
        # convert the torch tensor into a numpy array
        probs = probs.data.cpu().numpy()
        return [decode(p, beam_size=3, blank=self.blank)[0]
                    for p in probs]
        
    def infer_maxdecode(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        probs, rnn_args = self.forward_impl(x, softmax=True)
        # convert the torch tensor into a numpy array
        probs = probs.data.cpu().numpy()
        return [decode(p, blank=self.blank) for p in probs]
    
    def infer_distribution(self, batch, num_results):
        x, y, x_lens, y_lens = self.collate(*batch)
        probs, rnn_args = self.forward_impl(x, softmax=True)
        probs = probs.data.cpu().numpy()
        return [decode_dist(p, beam_size=3, blank=self.blank)
                    for p in probs]

    @staticmethod
    def max_decode(pred, blank):
        prev = pred[0]
        seq = [prev] if prev != blank else []
        for p in pred[1:]:
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        return seq

