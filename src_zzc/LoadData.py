import os
import sys
import random
import math
import numpy as np
import pickle


class DataItem(object):
    def __init__(self, msg):
        """
            Init
            \param msg: data
            \param gt: ground truth
        """
        self.msg = msg
        self.gt = None

    def set_gt(self, gt):
        self.gt = gt

    def __iter__(self):
        return iter((self.gt, self.msg))

    def __repr__(self):
        return f"gt: {self.gt} | msg: {self.msg}"


class DataLoader(object):
    CROSS_VALID = 'cross_validation'
    TRAIN = 'train'
    TRAIN_USE_ALL = 'train_using_all'
    TRAIN_SUBSET = 'train_using_subset'

    def __init__(self, argv):
        # seed = random.randrange(sys.maxsize)
        random.seed(argv.seed)
        print("Seed was:", argv.seed)

        self.data_root = argv.data_root
        self.fold = argv.fold
        self.batch_size = argv.batch_size
        self.desired_len_percent = argv.desired_len_percent
        self.desired_len = None

        self.data_train = []
        self.data_test = []
        self.train_data = []
        self.test_data = []
        self.prepared_train_data = []
        self.prepared_test_data = []
        self.word2idx = None
        self.labels = set()
        self.prepared_train_data_distro = dict()
        self.prepared_test_data_distro = dict()
        self.prepared_train_data_weights = None
        self.max_len = 0
        self.min_len = sys.maxsize
        self.glove = None

    def read_data(self, mode):
        """
            Read data in
        """
        print(f"Reading data for {mode}")

        self.data_train = []
        self.data_test = []
        with open(os.path.join(self.data_root, 'Training.txt'),
                  "r",
                  encoding="utf-8") as f:
            for line in f:
                msg = line.strip().split(',')
                msg = [int(item) for item in msg]
                self.data_train.append(DataItem(msg))

        with open(os.path.join(self.data_root, 'Training_Label.txt'),
                  "r",
                  encoding="utf-8") as f:
            for i, line in enumerate(f):
                gt = line.strip()
                gt = int(gt)
                self.data_train[i].set_gt(gt)
                self.labels.add(gt)

        random.shuffle(self.data_train)
        if mode == self.TRAIN_USE_ALL:
            with open(os.path.join(self.data_root, 'Test.txt'),
                      "r",
                      encoding="utf-8") as f:
                for line in f:
                    msg = line.strip().split(',')
                    self.data_test.append(DataItem(msg))
        else:
            pivot = int(len(self.data_train) / self.fold)
            self.data_test = self.data_train[:pivot]
            self.data_train = self.data_train[pivot:]

    def get_batch(self, train=False):
        """
            Get data in batchs for DNN
        """
        if train:
            return self._get_batch_helper(self.prepared_train_data)
        else:
            return self._get_batch_helper(self.prepared_test_data)

    def _get_batch_helper(self, data):
        for i in range(math.ceil(len(data) / self.batch_size)):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            if end < len(data):
                batch = data[start:end]
            else:
                batch = data[start:]
                for i in range(self.batch_size - len(batch)):
                    batch.append(data[i])

                # while len(batch) < self.batch_size:
                #     batch.append(random.choice(data))
            # list of gt + list of msg idx, ordered by len(msg)
            batch.sort(key=lambda x: x.msg.count(self.word2idx["<PAD>"]))
            yield list(zip(*batch)) + [[
                len(item.msg) - item.msg.count(self.word2idx["<PAD>"])
                for item in batch
            ]]

    def prepare_data(self):
        """
            Prepare the data into index form, pad if undersized, truncate if oversized
        """
        # init word2idx mapping
        if self.word2idx == None:
            corpus = [word for gt, msg in self.train_data for word in msg]
            vocab = sorted(set(corpus))
            self.word2idx = {"<PAD>": 0}
            for word in vocab:
                if self.word2idx.get(word) is None:
                    if self.glove is None or word in self.glove.keys():
                        self.word2idx[word] = len(self.word2idx)
            if self.glove is not None:
                for word in sorted(self.glove.keys()):
                    if word not in self.word2idx.keys():
                        self.word2idx[word] = len(self.word2idx)

        # prepare the data into idxs
        data = self.train_data + self.test_data
        data_split_pos = math.floor(0.8 * len(data))
        self.desired_len = int(self.min_len + self.desired_len_percent *
                               (self.max_len - self.min_len))
        for gt, msg in data:
            idx = []

            for word in msg:
                # TODO: Add Unknown word support
                if word in self.word2idx.keys():
                    idx.append(self.word2idx[word])

            if len(idx) == 0:
                continue
            elif len(idx) < self.desired_len:
                idx.extend([0] * (self.desired_len - len(idx)))
            else:
                idx = idx[:self.desired_len]

            gt = self.labels.index(gt)

            if len(self.prepared_train_data) < data_split_pos:
                self.prepared_train_data.append(DataItem(gt, idx))
                self.prepared_train_data_distro[gt] = self.prepared_train_data_distro.get(gt, 0) + 1 # yapf: disable
            else:
                self.prepared_test_data.append(DataItem(gt, idx))
                self.prepared_test_data_distro[gt] = self.prepared_test_data_distro.get(gt, 0) + 1 # yapf: disable

        # compute the weight for losses of each classes
        self.train_data_weights = [
            max(self.prepared_train_data_distro.values()) /
            self.prepared_train_data_distro[self.labels.index(key)]
            for key in self.labels
        ]