import os
import sys
import operator
import math
import pickle

NB_MODEL_PARAM_PATH = "NB_param.pkl"


class NB_Model(object):
    def __init__(self, argv, eps=0.1):
        self.argv = argv
        self.labels = list(range(1, argv.num_label + 1))
        self.wc = dict.fromkeys(self.labels)
        for key in self.labels:
            self.wc[key] = dict()
        self.c = dict.fromkeys(self.labels, 0)
        self.unknown = dict.fromkeys(self.labels)
        self.vocab = set()

        self.eps = eps
        self.unknown_cnt = 0
        self.known_cnt = 0

    def train(self, data):
        total_word = dict.fromkeys(self.labels, 0)

        # cnt
        for c, msg in data:
            self.c[c] += 1
            total_word[c] += len(msg)
            for word in msg:
                self.vocab.add(word)
                if word not in self.wc[c].keys():
                    self.wc[c][word] = 1
                else:
                    self.wc[c][word] += 1

        # compute p(w|c) and p(c) and p(unknown) with Laplace smoothing
        for c in self.wc.keys():
            total = total_word[c] + len(self.wc[c].keys()) * self.eps
            self.c[c] /= len(data)
            self.unknown[c] = self.eps / total
            for word in self.wc[c].keys():
                self.wc[c][word] = (self.wc[c][word] + self.eps) / total

    def test(self, data):
        preds = []
        self.unknown_cnt = 0
        self.known_cnt = 0
        for _, msg in data:
            predict = dict.fromkeys(self.labels)
            for c in predict.keys():
                predict[c] = math.log(self.c[c])
                for word in msg:
                    if word in self.wc[c].keys():
                        predict[c] += math.log(self.wc[c][word], 2)
                        self.known_cnt += 1
                    else:
                        predict[c] += math.log(self.unknown[c], 2)
                        self.unknown_cnt += 1
            result = max(predict.items(), key=operator.itemgetter(1))[0]
            preds.append(result)
        return preds

    def state_dict(self):
        return {
            "wc": self.wc,
            "c": self.c,
            "unknown": self.unknown,
            "vocab": self.vocab,
            "eps": self.eps,
            "unknown_cnt": self.unknown_cnt,
            "known_cnt": self.known_cnt
        }

    def load_state_dict(self, checkpoint):
        self.wc = checkpoint["wc"]
        self.c = checkpoint["c"]
        self.unknown = checkpoint["unknown"]
        self.vocab = checkpoint["vocab"]
        self.eps = checkpoint["eps"]
        self.unknown_cnt = checkpoint["unknown_cnt"]
        self.known_cn = checkpoint["known_cnt"]
