import os
import sys
import operator
import math
import pickle

NB_MODEL_PARAM_PATH = "NB_param.pkl"


class NB_Model(object):
    def __init__(self, argv, labels, eps=0.1):
        self.argv = argv
        self.labels = labels
        self.wc = dict.fromkeys(labels)
        for key in labels:
            self.wc[key] = dict()
        self.c = dict.fromkeys(labels, 0)
        self.unknown = dict.fromkeys(labels)
        self.vocab = set()

        self.eps = eps

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
        for gt, msg in data:
            predict = dict.fromkeys(self.labels)
            for c in predict.keys():
                predict[c] = math.log(self.c[c])
                for word in msg:
                    if word in self.wc[c].keys():
                        predict[c] += math.log(self.wc[c][word], 2)
                    else:
                        predict[c] += math.log(self.unknown[c], 2)
            result = max(predict.items(), key=operator.itemgetter(1))[0]
            preds.append(result)
        return preds