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
        self.gt = 0

    def set_gt(self, gt):
        self.gt = gt

    def __iter__(self):
        return iter((self.gt, self.msg))

    def __repr__(self):
        return f"gt: {self.gt} | msg: {self.msg}"


class DataLoader(object):
    TRAIN_USE_ALL = 'train_using_all'
    TRAIN_WITH_VALIDATION = 'train_with_val'

    def __init__(self, argv):
        # seed = random.randrange(sys.maxsize)
        random.seed(argv.seed)
        print("Seed was:", argv.seed)

        self.data_root = argv.data_root
        self.fold = argv.fold
        self.batch_size = argv.batch_size
        self.desired_len_percent = argv.desired_len_percent
        self.desired_len = None

        self.labels = set()
        self.data_train = []
        self.data_val = []
        self.data_test = []

        self.train_data_distro = dict()
        self.train_data_weights = None
        self.len_max = 0
        self.len_min = sys.maxsize
        self.idx_max = 0

    def read_data(self, mode):
        """
            Read data in
        """
        print(f"Reading data for {mode}")

        self.data_train = []
        self.data_val = []
        self.data_test = []
        with open(os.path.join(self.data_root, 'Training.txt'),
                  "r",
                  encoding="utf-8") as f:
            for line in f:
                msg = line.strip().split(',')
                msg = [int(item) for item in msg]
                self.data_train.append(DataItem(msg))
                self.len_max = len(msg) if len(msg) > self.len_max else self.len_max # yapf: disable
                self.len_min = len(msg) if len(msg) < self.len_min else self.len_min # yapf: disable
                idx_max = max(msg)
                self.idx_max = idx_max if idx_max > self.idx_max else self.idx_max # yapf: disable

        with open(os.path.join(self.data_root, 'Training_Label.txt'),
                  "r",
                  encoding="utf-8") as f:
            for i, line in enumerate(f):
                gt = line.strip()
                gt = int(gt)
                self.data_train[i].set_gt(gt)
                self.labels.add(gt)
                self.train_data_distro[gt] = self.train_data_distro.get(gt, 0) + 1 # yapf: disable

        # compute the weight for losses of each classes
        self.train_data_weights = [
            max(self.train_data_distro.values()) / self.train_data_distro[key]
            for key in self.labels
        ]
        self.desired_len = int(self.len_min + self.desired_len_percent *
                               (self.len_max - self.len_min))

        random.shuffle(self.data_train)
        if mode == self.TRAIN_WITH_VALIDATION:
            pivot = int(len(self.data_train) / self.fold)
            self.data_val = self.data_train[:pivot]
            self.data_train = self.data_train[pivot:]

        with open(os.path.join(self.data_root, 'Test.txt'),
                  "r",
                  encoding="utf-8") as f:
            for line in f:
                msg = line.strip().split(',')
                msg = [int(item) for item in msg]
                self.data_test.append(DataItem(msg))

    def get_batch(self, train=False, val=False):
        """
            Get data in batchs for DNN
        """
        if train:
            return self._get_batch_helper(self.data_train)
        elif val:
            return self._get_batch_helper(self.data_val)
        else:
            return self._get_batch_helper(self.data_test)

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

            for item in batch:
                if len(item.msg) < self.desired_len:
                    item.msg.extend([0] * (self.desired_len - len(item.msg)))
                else:
                    item.msg = item.msg[:self.desired_len]

            # list of gt + list of msg idx, ordered by len(msg)
            batch.sort(key=lambda x: x.msg.count(0))
            yield list(zip(*batch)) + [[
                len(item.msg) - item.msg.count(0) for item in batch
            ]]

    def prepare_data(self):
        """
            Prepare the data into index form, pad if undersized, truncate if oversized
        """
        # prepare the data into idxs
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
