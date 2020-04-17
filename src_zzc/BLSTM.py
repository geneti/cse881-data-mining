import os
import string
import sys
import argparse
import numpy as np
import random
import pprint
import math
import pickle
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class BLSTM(nn.Module):
    def __init__(self, argv, desired_len):
        super(BLSTM, self).__init__()
        self.pad = 0
        self.argv = argv
        self.desired_len = desired_len

        self.emb_dropout = nn.Dropout(argv.emb_drop_r)

        self.embed = nn.Embedding(argv.max_idx + 1,
                                  argv.emb_size,
                                  padding_idx=self.pad)

        self.lstm = nn.LSTM(
            argv.emb_size,
            argv.lstm_hidden_sz,
            argv.lstm_n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=argv.lstm_drop_r if argv.lstm_n_layer > 1 else 0)

        self.fc = nn.Sequential(
            nn.Linear(argv.lstm_hidden_sz, argv.num_label),
            nn.Softmax(dim=1),
        )

    def init_weight(self):
        self.embed.weight = nn.init.xavier_uniform_(self.embed.weight)
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)

    def init_hidden(self):
        forward = Variable(
            torch.zeros(self.argv.lstm_n_layer * 2, self.argv.batch_size,
                        self.argv.lstm_hidden_sz))
        backward = Variable(
            torch.zeros(self.argv.lstm_n_layer * 2, self.argv.batch_size,
                        self.argv.lstm_hidden_sz))
        return ((forward.cuda(),
                 backward.cuda()) if next(self.parameters()).is_cuda else
                (forward, backward))

    def forward(self, msgs, msg_len, is_training=False):
        batch_size = msgs.size(0)
        hidden = self.init_hidden()

        embeds = self.embed(msgs)
        if is_training:
            embeds = self.emb_dropout(embeds)

        # Run BLSTM
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds,
                                                         msg_len,
                                                         batch_first=True)
        out, _ = self.lstm(embeds, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = out[:, :, :self.argv.lstm_hidden_sz] + out[:, :, self.argv.lstm_hidden_sz:] # yapf: disable
        h = out.sum(1) / msg_len.unsqueeze(-1)

        # h = out.gather(1, (msg_len - 1).unsqueeze(-1).unsqueeze(-1).expand(
        #     batch_size, 1, self.argv.lstm_hidden_sz)).squeeze(1)
        # h += out[:, 0, :]
        # h /= 2

        res = self.fc(h)
        return res
