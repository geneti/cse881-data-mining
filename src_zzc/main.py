import os
import sys
import numpy as np
import time
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import utils
from LoadData import DataLoader
from NaiveBayes import NB_Model
# from BLSTM_2DCNN import BLSTM_2DCNN

MODE = ["train", "test", "all"]
MODEL = ["nb", "blstm", "all"]
ACTIVATION_FUNC = ["ReLU", "Tanh"]
OPTIMIZER = ["Adam", "SGD"]

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def parse_arguments(argv):
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str.lower, help="whether to train or test the model.",
                        default="test", choices=MODE, )
    parser.add_argument("model", type=str.lower, help="which model to use",
                        choices=MODEL, default="E2ENCR", )
    parser.add_argument("--data_root", type=str,
                        help="path to the data",
                        default=".\\dataset", )
    parser.add_argument("--res_root", type=str,
                        help="root for the results",
                        default=".\\results", )
    parser.add_argument("--fold", type=int, help="number of fold", default=10)
    parser.add_argument("--checkpoint_root", type=str,
                        help="path to the checkpoint",
                        default="./checkpoints", )
    parser.add_argument("--seed", type=int, help="seed used for randomness", default=int(time.time()))

    # BLSTM-2DCNN
    parser.add_argument("--use_checkpoint",
                        help="whether or not to use checkpoint",
                        action="store_true",)
    parser.add_argument("--checkpoint_name", type=str,
                        help="checkpoint file name",
                        default='checkpoint.pkl')
    parser.add_argument("--epoch", type=int, help="number of epoch", default=50)
    parser.add_argument("--batch_size", type=int, help="number of epoch", default=20)
    parser.add_argument("--optimizer", type=str, help="type of optimizer function",
                        default="Adam", choices=OPTIMIZER, )
    parser.add_argument("--activate_func", type=str, help="type of activation function",
                        default="ReLU", choices=ACTIVATION_FUNC,)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    parser.add_argument("--is_glove",
                        help="whether or not to use pretrained glove embedding",
                        action="store_true",)
    parser.add_argument("--glove_path", type=str,
                        help="path to the glove folder that contains the pretrained embedding",
                        default="./glove",)
    parser.add_argument("--desired_len_percent", type=float,
                        help="the percentage of padding for each message", default=1.0,)

    # Embed Layer
    parser.add_argument("--emb_drop_r", type=float, help="embedding drop rate", default=0.5)
    parser.add_argument("--emb_size", type=int,
                        help="embedding size", default=300)
    # LSTM Layer
    parser.add_argument("--lstm_drop_r", type=float, help="embedding drop rate", default=0.2)
    parser.add_argument("--lstm_n_layer", type=int,
                        help="number of hidden layers in lstm", default=1)
    parser.add_argument("--lstm_hidden_sz", type=int,
                        help="hidden layer size for lstm", default=300)

    # CNN layer
    parser.add_argument("--cnn_n_kernel", type=int,
                        help="number of kernel/filter for cnn", default=100)
    parser.add_argument("--cnn_kernel_sz", type=int,
                        help="kernel size for cnn", default=3)
    parser.add_argument("--cnn_pool_kernel_sz", type=int,
                        help="kernel size for pooling layer", default=2)
    # yapf: enable

    argv = parser.parse_args(argv)
    return argv


def train_blstm_2dcnn(argv, model, loader):
    loss_func = nn.CrossEntropyLoss(FloatTensor(loader.train_data_weights))
    param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = (optim.Adam(param, lr=argv.lr)
                 if argv.optimizer == "Adam" else optim.SGD(param, lr=argv.lr))
    if argv.use_checkpoint and os.path.exists(
            os.path.join(argv.checkpoint_root, argv.checkpoint_name)):
        print(
            f"Training: Loading {os.path.join(argv.checkpoint_root, argv.checkpoint_name)}"
        )
        checkpoint = torch.load(
            os.path.join(argv.checkpoint_root, argv.checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_offset = checkpoint['epoch']
    else:
        model.init_weight(loader.glove)
        epoch_offset = 0

    for epoch in range(argv.epoch):
        epoch += epoch_offset
        losses = []
        acc = []
        for i, batch in enumerate(loader.get_batch(train=True)):
            gts = LongTensor(batch[0])
            msgs = LongTensor(batch[1])
            msg_len = LongTensor(batch[2])

            model.zero_grad()
            preds = model.forward(msgs, msg_len, is_training=True)
            loss = loss_func(preds, gts)

            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)  # gradient clipping
            optimizer.step()

            acc.append((torch.max(preds, 1)[1] == gts).sum().item() * 100 /
                       msgs.size(0))
        print(
            f"\t[{epoch:02d}/{argv.epoch+epoch_offset}] train: loss : {np.mean(losses):0.2f}, acc: {np.mean(acc):0.2f}% | ",
            end='')
        test_blstm_2dcnn(model, loader)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(argv.checkpoint_root, argv.checkpoint_name))


def test_blstm_2dcnn(model, loader, only=False):
    loss_func = nn.CrossEntropyLoss(FloatTensor(loader.train_data_weights))
    test_losses = []
    preds_list = []
    gts_list = []
    for batch in loader.get_batch():
        gts = LongTensor(batch[0])
        msgs = LongTensor(batch[1])
        msg_len = LongTensor(batch[2])

        with torch.no_grad():
            model.zero_grad()
            preds = model.forward(msgs, msg_len, is_training=False)
            test_losses.append(loss_func(preds, gts).item())

            gts_list.append(gts.cpu())
            preds_list.append(preds.cpu())

    preds = torch.max(torch.cat(preds_list, 0), 1)[1]
    gts = torch.cat(gts_list, 0)
    loss = np.mean(test_losses)
    acc = accuracy_score(gts, preds)
    print(f"Test: mean_loss : {loss:0.2f}, acc: {acc*100:0.2f}%")
    if only:
        print(classification_report(gts, preds, target_names=loader.labels))
        plot_confusion_matrix(loader, gts, preds, "BLSTM-2DCNN.png")


def main(argv):
    loader = DataLoader(argv)
    loader.read_data(DataLoader.TRAIN_SUBSET)

    if not os.path.exists(argv.res_root):
        os.mkdir(argv.res_root)

    if argv.model in ["nb", "all"]:
        print(f"##### Naive Bayes {'#'*50}")
        NB_model = NB_Model(argv, loader.labels)

        if argv.mode in ["train", "test", "all"]:
            # train Naive Bayes
            NB_model.train(loader.data_train)
            # test Naive Bayes
            preds = NB_model.test(loader.data_test)
            gts = list(zip(*loader.data_test))[0]
            print(f'\tThe accuracy is %.2f%%' %
                  (accuracy_score(gts, preds) * 100))
            print(
                classification_report(
                    gts, preds, target_names=[str(l) for l in loader.labels]))
            utils.plot_confusion_matrix(
                loader, gts, preds, os.path.join(argv.res_root, "Bayes.png"))
            out_path = os.path.join(argv.res_root, 'Bayes.txt')
            utils.save_preds(out_path, preds)
            # print(incorrect)

    # if argv.model in ["blstm", "all"]:
    #     print(f"##### BLSTM-2DCNN {'#'*50}")
    #     if argv.is_glove:
    #         loader.read_glove(argv.glove_path, argv.emb_size)
    #     loader.prepare_data()

    #     blstm_2dcnn_model = BLSTM_2DCNN(argv, loader.word2idx,
    #                                     loader.desired_len)

    #     if USE_CUDA:
    #         blstm_2dcnn_model = blstm_2dcnn_model.cuda()
    #     if argv.mode in ["train", "all"]:
    #         # train BLSTM-2DCNN
    #         train_blstm_2dcnn(argv, blstm_2dcnn_model, loader)
    #     if argv.mode in ["test", "all"]:
    #         # test BLSTM-2DCNN
    #         checkpoint = torch.load(
    #             os.path.join(argv.checkpoint_root, argv.checkpoint_name))
    #         blstm_2dcnn_model.load_state_dict(checkpoint['model_state_dict'])
    #         test_blstm_2dcnn(blstm_2dcnn_model, loader, only=True)

    print()


if __name__ == "__main__":
    # main(parse_arguments(sys.argv[1:]))
    main(
        parse_arguments("all all "
                        "--data_root .\\dataset "
                        "--fold 10 "
                        # "--use_checkpoint "
                        "--checkpoint_name checkpoint_acc98_v2.pkl "
                        # "--seed 4437098522973987586 "
                        "--epoch 10 "
                        "--batch_size 50 "
                        "--optimizer Adam "
                        "--activate_func ReLU "
                        "--lr 0.001 "
                        # "--is_glove "
                        "--desired_len_percent 0.5 "
                        "--emb_drop_r 0.5 "
                        "--emb_size 300 "
                        "--lstm_drop_r 0.2 "
                        "--lstm_n_layer 1 "
                        "--lstm_hidden_sz 300 "
                        "--cnn_n_kernel 50 "
                        "--cnn_kernel_sz 3 "
                        "--cnn_pool_kernel_sz 2 ".split()))
