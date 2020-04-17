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
        random.seed(argv.seed)
        print("DataLoader Seed was:", argv.seed)

        self.argv = argv
        self.data_path = argv.data_path
        self.fold = argv.fold
        self.batch_size = argv.batch_size
        self.desired_len_percent = argv.desired_len_percent
        self.mode = argv.mode
        self.desired_len = None

        self.data = []
        self.data_val = []

        self.data_distro = dict()
        self.data_weights = None
        self.len_max = 0
        self.len_min = sys.maxsize
        self.max_idx = 0

        self.label_is_val = False

    def read_data(self):
        """
            Read data in
        """
        print(f"Reading data from {self.data_path}")

        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                msg = line.strip().split(',')
                msg = [int(item) for item in msg]
                self.data.append(DataItem(msg))
                self.len_max = len(msg) if len(msg) > self.len_max else self.len_max # yapf: disable
                self.len_min = len(msg) if len(msg) < self.len_min else self.len_min # yapf: disable
                max_idx = max(msg)
                self.max_idx = max_idx if max_idx > self.max_idx else self.max_idx # yapf: disable

        f_name, f_ext = os.path.splitext(self.data_path)
        label_path = f"{f_name}_Label{f_ext}"
        if os.path.exists(label_path):
            self.label_is_val = True
            print(f"Reading data label from {label_path}")
            with open(label_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    gt = line.strip()
                    gt = int(gt)
                    self.data[i].set_gt(gt)
                    self.data_distro[gt] = self.data_distro.get(gt, 0) + 1 # yapf: disable

        if self.mode == 'train':
            random.shuffle(self.data)
            pivot = int(len(self.data) / self.fold)
            self.data_val = self.data[:pivot]
            self.data = self.data[pivot:]
            for data in self.data_val:
                self.data_distro[data.gt] -= 1

            # compute the weight for losses of each classes
            self.data_weights = [
                max(self.data_distro.values()) / self.data_distro[key]
                for key in list(range(1, self.argv.num_label + 1))
            ]
            self.desired_len = int(self.len_min + self.desired_len_percent *
                                (self.len_max - self.len_min))

    def get_batch(self, val=False):
        """
            Get data in batchs for DNN
        """
        if not val:
            return self._get_batch_helper(self.data)
        else:
            return self._get_batch_helper(self.data_val)

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

            for item in batch:
                if len(item.msg) < self.desired_len:
                    item.msg.extend([0] * (self.desired_len - len(item.msg)))
                else:
                    item.msg = item.msg[:self.desired_len]

            # list of gt + list of msg idx, ordered by len(msg)
            # batch.sort(key=lambda x: x.msg.count(0))
            yield list(zip(*batch)) + [[
                len(item.msg) - item.msg.count(0) for item in batch
            ]]

    # def prepare_data(self):
    #     """
    #         Prepare the data into index form, pad if undersized, truncate if oversized
    #     """
    #     # prepare the data into idxs
    #     for gt, msg in data:
    #         idx = []

    #         for word in msg:
    #             # TODO: Add Unknown word support
    #             if word in self.word2idx.keys():
    #                 idx.append(self.word2idx[word])

    #         if len(idx) == 0:
    #             continue
    #         elif len(idx) < self.desired_len:
    #             idx.extend([0] * (self.desired_len - len(idx)))
    #         else:
    #             idx = idx[:self.desired_len]
