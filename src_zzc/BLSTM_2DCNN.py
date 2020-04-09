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


class BLSTM_2DCNN(nn.Module):
    def __init__(self, argv, desired_len, idx_max, labels):
        super(BLSTM_2DCNN, self).__init__()
        self.pad = 0
        self.argv = argv
        self.labels = labels
        self.desired_len = desired_len

        self.emb_dropout = nn.Dropout(argv.emb_drop_r)

        self.embed = nn.Embedding(idx_max + 1,
                                  argv.emb_size,
                                  padding_idx=self.pad)

        self.lstm = nn.LSTM(
            argv.emb_size,
            argv.lstm_hidden_sz,
            argv.lstm_n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=argv.lstm_drop_r if argv.lstm_n_layer > 1 else 0)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, argv.cnn_n_kernel, kernel_size=argv.cnn_kernel_sz),
        #     nn.ReLU() if argv.activate_func == "ReLU" else nn.Tanh(),
        #     nn.MaxPool2d(kernel_size=argv.cnn_pool_kernel_sz),
        # )

        # self.hidden_sz_after_cnn = int(
        #     (argv.lstm_hidden_sz - argv.cnn_kernel_sz + 1) /
        #     argv.cnn_pool_kernel_sz)
        # self.len_after_cnn = int(
        #     (desired_len - argv.cnn_kernel_sz + 1) / argv.cnn_pool_kernel_sz)


        self.fc = nn.Sequential(
            nn.Linear(argv.lstm_hidden_sz, len(labels)),
            nn.Softmax(dim=1),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(argv.lstm_hidden_sz, len(labels)),
        #     nn.Softmax(dim=1),
        # )

    def init_weight(self):
        self.embed.weight = nn.init.xavier_uniform_(self.embed.weight)
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)

    # def _get_pretrained_emb_weights(self, glove, word2idx):
    #     emb_size = len(next(iter(glove.values())))
    #     matrix = np.zeros((len(word2idx), emb_size))
    #     for word in word2idx.keys():
    #         try:
    #             matrix[word2idx[word]] = glove[word]
    #         except KeyError:
    #             # if word != '<PAD>':
    #             matrix[word2idx[word]] = np.random.normal(scale=0.6,
    #                                                       size=(emb_size, ))
    #     return torch.from_numpy(matrix)

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

        h = out[:, :, :self.argv.lstm_hidden_sz] + out[:, :, self.argv.lstm_hidden_sz:] # yapf: disable
        h = h.sum(1) / msg_len

        # h = h.unsqueeze(1)

        # CNN layer
        # O = self.conv(h)
        # O = F.pad(O, (0, 0, 0, self.len_after_cnn - O.size(2)),
        #           mode="constant",
        #           value=0)

        # O = O.view(-1, self.fc[0].in_features)
        res = self.fc(h)
        # h = h.sum(1) / msg_len.unsqueeze(1)
        # res = self.fc2(h)
        return res
