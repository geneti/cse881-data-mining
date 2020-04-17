import os
import sys
import argparse
import numpy as np
import random
import pprint
import math
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


class DAN(nn.Module):
    def __init__(self, argv):
        super(DAN, self).__init__()
        self.pad = 0
        self.argv = argv

        self.emb_dropout = nn.Dropout(argv.emb_drop_r)

        self.embed = nn.Embedding(argv.max_idx + 1,
                                  argv.emb_size,
                                  padding_idx=self.pad)
        hidden_layers = [nn.Linear(argv.emb_size, argv.dan_hidden_sz)]
        for _ in range(argv.dan_num_hidden):
            if argv.activate_func == 'ReLU':
                hidden_layers.append(nn.ReLU())
            elif argv.activate_func == 'Tanh':
                hidden_layers.append(nn.Tanh())
            hidden_layers.append(
                nn.Linear(argv.dan_hidden_sz, argv.dan_hidden_sz))
        hidden_layers[-1] = nn.Linear(argv.dan_hidden_sz, argv.num_label)
        self.hiddens = nn.Sequential(*hidden_layers)
        self.softmax = nn.Softmax(dim=1)

    def init_weight(self):
        self.embed.weight = nn.init.xavier_uniform(self.embed.weight)
        for name, param in self.hiddens.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform(param)

    def forward(self, msgs, msg_len, is_training=False):
        embeds = self.embed(msgs)

        embeds[(msgs == self.pad)] = 0  # 0 out all padding embedding
        if is_training:
            embeds = self.emb_dropout(embeds)


        embeds = embeds.sum(dim=1) / msg_len.unsqueeze(-1).float()

        hidden = self.hiddens(embeds)
        out = self.softmax(hidden)
        return out