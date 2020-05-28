from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.autograd as autograd

from . import model
from .ctc_decoder import decode
from .ctc_decoder_dist import decode_dist


class CTC(model.Model):
    def __init__(self, freq_dim, output_dim, config):
        super().__init__(freq_dim, config)

        # include the blank token
        self.blank = output_dim
        self.fc = model.LinearND(self.encoder_dim, output_dim + 1)

    def forward(self, x, rnn_args=None):
       # x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_impl(x, rnn_args,  softmax=True)

    def forward_impl(self, x, rnn_args=None, softmax=False):
        if self.is_cuda:
            x = x.cuda()
        x, rnn_args = self.encode(x, rnn_args)    
        x = self.fc(x)          
        if softmax:
            return torch.nn.functional.softmax(x, dim=2), rnn_args
        return x, rnn_args

    def loss(self, batch):
        pass

    def collate(self, inputs, labels):
        max_t = max(i.shape[0] for i in inputs)
        max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        x = torch.FloatTensor(model.zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]

        if self.volatile:
            for v in batch:
                v.volatile = True
        return batch
    
    def infer(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        probs, rnn_args = self.forward_impl(x, softmax=True)
        # convert the torch tensor into a numpy array
        probs = probs.data.cpu().numpy()
        return [decode(p, beam_size=3, blank=self.blank)[0]
                    for p in probs]
    def infer_confidence(self, batch):
        """
        returns the confidence value was well as the prediction
        as a tuple
        """
        x, y, x_lens, y_lens = self.collate(*batch)
        probs, rnn_args = self.forward_impl(x, softmax=True)
        # convert the torch tensor into a numpy array
        probs = probs.data.cpu().numpy()
        preds_confidence = [decode(p, beam_size=3, blank=self.blank)
                    for p in probs]
        preds = [x[0] for x in preds_confidence]
        confidence = [x[1] for x in preds_confidence]
        return preds, confidence    
    
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
