import os
import sys
import numpy as np
import time
import math
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
from BLSTM_2DCNN import BLSTM_2DCNN

MODE = ["train", "test", "all"]
MODEL = ["nb", "blstm", "all"]
ACTIVATION_FUNC = ["ReLU", "Tanh"]
OPTIMIZER = ["Adam", "SGD"]

USE_CUDA = False
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def parse_arguments(argv):
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",         default="test", type=str.lower, choices=MODE, help="whether to train or test the model.",)
    parser.add_argument("model",        default="E2ENCR", type=str.lower,  choices=MODEL, help="which model to use",)
    parser.add_argument("--gpu_id",       default=0, type=int, help="gpu id")
    parser.add_argument("--data_root",  default="./src_zzc/data", type=str, help="path to the data",)
    parser.add_argument("--res_root",   default="./src_zzc/results", type=str, help="root for the results",)
    parser.add_argument("--fold",       default=10, type=int, help="number of fold")
    parser.add_argument("--checkpoint_root", default="./src_zzc/checkpoints", type=str, help="path to the checkpoint",)
    parser.add_argument("--seed",       default=int(time.time()), type=int, help="seed used for randomness",)

    # BLSTM-2DCNN
    parser.add_argument("--use_checkpoint",     action="store_true",    help="whether or not to use checkpoint",)
    parser.add_argument("--checkpoint_name",    default='checkpoint',   type=str, help="checkpoint file name", )
    parser.add_argument("--checkpoint_ver",     default=0,              type=int, help="checkpoint version")
    parser.add_argument("--epoch",              default=50,             type=int, help="number of epoch")
    parser.add_argument("--batch_size",         default=20,             type=int, help="number of epoch")
    parser.add_argument("--lr",                 default=0.01,           type=float, help="learning rate")
    parser.add_argument("--desired_len_percent", default=1.0,           type=float, help="the percentage of padding for each message")
    parser.add_argument("--optimizer",          default="Adam",         choices=OPTIMIZER, type=str, help="type of optimizer function", )
    parser.add_argument("--activate_func",      default="ReLU",         choices=ACTIVATION_FUNC, type=str, help="type of activation function",)

    # Embed Layer
    parser.add_argument("--emb_drop_r", default=0.5, type=float, help="embedding drop rate")
    parser.add_argument("--emb_size",   default=300, type=int,   help="embedding size")
    # LSTM Layer
    parser.add_argument("--lstm_drop_r",    default=0.2, type=float, help="embedding drop rate")
    parser.add_argument("--lstm_n_layer",   default=1,   type=int,   help="number of hidden layers in lstm")
    parser.add_argument("--lstm_hidden_sz", default=300, type=int,   help="hidden layer size for lstm")

    # CNN layer
    parser.add_argument("--cnn_n_kernel",       default=100, type=int, help="number of kernel/filter for cnn")
    parser.add_argument("--cnn_kernel_sz",      default=3,   type=int, help="kernel size for cnn")
    parser.add_argument("--cnn_pool_kernel_sz", default=2,   type=int, help="kernel size for pooling layer")
    # yapf: enable

    argv = parser.parse_args(argv)
    argv.model_folder = f"f{argv.fold}_bs{argv.batch_size}_es{argv.emb_size}_hs{argv.lstm_hidden_sz}"
    return argv


def valid_blstm_2dcnn(model, loader, epoch, only=False):
    loss_func = nn.CrossEntropyLoss(FloatTensor(loader.train_data_weights))
    test_losses = []
    preds_list = []
    gts_list = []
    for batch in loader.get_batch(train=False, val=True):
        gts = LongTensor(batch[0]) - 1
        msgs = LongTensor(batch[1])
        msg_len_ordered, msg_perm = (msgs != model.pad).sum(dim=1).sort(descending=True) # yapf: disable
        msgs = msgs[msg_perm]
        gts = gts[msg_perm]

        with torch.no_grad():
            model.zero_grad()
            preds = model.forward(msgs, msg_len_ordered, is_training=False)
            test_losses.append(loss_func(preds, gts).item())

            gts_list.append(gts.cpu())
            preds_list.append(preds.cpu())

    preds = torch.max(torch.cat(preds_list, 0), 1)[1]
    gts = torch.cat(gts_list, 0)
    loss = np.mean(test_losses)
    acc = accuracy_score(gts, preds)
    print(f"Validation: mean_loss : {loss:0.2f}, acc: {acc*100:0.2f}%")
    if only:
        print(classification_report(gts, preds, target_names=loader.labels))
        out_path = os.path.join(model.argv.res_root, model.argv.model_folder,
                                f'BLSTM_validation_epoch{epoch}.png')
        utils.plot_confusion_matrix(loader, gts, preds, out_path)


def test_blstm_2dcnn(model, loader, epoch):
    preds_list = []
    for batch in loader.get_batch(train=False, val=False):
        msgs = LongTensor(batch[1])
        msg_len_ordered, msg_perm = (msgs != model.pad).sum(dim=1).sort(descending=True) # yapf: disable
        _, undo_msg_perm = msg_perm.sort()
        msgs = msgs[msg_perm]

        with torch.no_grad():
            model.zero_grad()
            preds = model.forward(msgs, msg_len_ordered, is_training=False)

            preds = preds[undo_msg_perm]
            preds_list.append(preds.cpu())

    preds = torch.max(torch.cat(preds_list, 0), 1)[1]
    out_path = os.path.join(model.argv.res_root, model.argv.model_folder,
                            f'BLSTM_2DCNN_test_epoch{epoch}.txt')
    utils.save_preds(out_path, preds[:len(loader.data_test)] + 1)


def train_blstm_2dcnn(argv, model, loader):
    loss_func = nn.CrossEntropyLoss(FloatTensor(loader.train_data_weights))
    param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = (optim.Adam(param, lr=argv.lr)
                 if argv.optimizer == "Adam" else optim.SGD(param, lr=argv.lr))

    checkpoint_path = os.path.join(argv.checkpoint_root, argv.model_folder,
                                   f'{argv.checkpoint_name}_epoch%d.pkl')
    chkpt_load_path = checkpoint_path % argv.checkpoint_ver
    if argv.use_checkpoint and os.path.exists(chkpt_load_path):
        print(f"Training: Loading {chkpt_load_path}")
        checkpoint = torch.load(chkpt_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_offset = checkpoint['epoch'] + 1
    else:
        model.init_weight()
        epoch_offset = 1

    for epoch in range(argv.epoch):
        epoch += epoch_offset
        losses = []
        acc = []
        for i, batch in enumerate(loader.get_batch(train=True, val=False)):
            gts = LongTensor(batch[0]) - 1
            msgs = LongTensor(batch[1])
            msg_len_ordered, msg_perm = (msgs != model.pad).sum(dim=1).sort(descending=True) # yapf: disable
            msgs = msgs[msg_perm]
            gts = gts[msg_perm]

            model.zero_grad()
            preds = model.forward(msgs, msg_len_ordered, is_training=True)
            loss = loss_func(preds, gts)

            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)  # gradient clipping
            optimizer.step()

            acc.append((torch.max(preds, 1)[1] == gts).sum().item() * 100 /
                       msgs.size(0))
            print(
                f"\r\t[{i+1:03d}/{math.ceil(len(loader.data_train) / argv.batch_size)}] "
                f"batch: loss : {np.mean(losses):0.2f}, acc: {np.mean(acc):0.2f}% | ",
                end='')
        print(
            f"\t[{epoch:02d}/{argv.epoch+epoch_offset}] "
            f"train: loss : {np.mean(losses):0.2f}, acc: {np.mean(acc):0.2f}% | ",
            end='')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path % epoch)
        valid_blstm_2dcnn(model, loader, epoch)
        test_blstm_2dcnn(model, loader, epoch)


def main(argv):
    torch.cuda.set_device(argv.gpu_id)
    print("Torch GPU set:", argv.gpu_id)
    torch.manual_seed(argv.seed)
    print("Torch Seed was:", argv.seed)
    utils.mkdir(os.path.join(argv.checkpoint_root, argv.model_folder))

    # train_mode = DataLoader.TRAIN_USE_ALL
    train_mode = DataLoader.TRAIN_WITH_VALIDATION

    loader = DataLoader(argv)
    loader.read_data(train_mode)

    if argv.model in ["nb", "all"]:
        print(f"##### Naive Bayes {'#'*50}")
        NB_model = NB_Model(argv, loader.labels)

        if argv.mode in ["train", "test", "all"]:
            # train Naive Bayes
            print('\tNB training')
            NB_model.train(loader.data_train)

            # validate Navie Bayes
            if len(loader.data_val) > 0:
                print('\tNB validation')
                preds = NB_model.test(loader.data_val)
                out_path = os.path.join(argv.res_root, 'Bayes_validation.txt')
                utils.save_preds(out_path, preds)
                gts = list(zip(*loader.data_val))[0]

                print(f'\tThe accuracy is %.2f%%' % (accuracy_score(gts, preds) * 100)) # yapf: disable
                target_names = [str(l) for l in loader.labels]
                print(
                    classification_report(gts,
                                          preds,
                                          target_names=target_names))
                utils.plot_confusion_matrix(
                    loader.labels, gts, preds,
                    os.path.join(argv.res_root, "Bayes_validation.png"))
            # test Naive Bayes
            print('\tNB testing')
            preds = NB_model.test(loader.data_test)

            out_path = os.path.join(argv.res_root, 'Bayes_test.txt')
            utils.save_preds(out_path, preds)

    if argv.model in ["blstm", "all"]:
        print(f"##### BLSTM-2DCNN {'#'*50}")

        blstm_2dcnn_model = BLSTM_2DCNN(argv, loader.desired_len,
                                        loader.idx_max, loader.labels)

        if USE_CUDA:
            blstm_2dcnn_model = blstm_2dcnn_model.cuda()
        if argv.mode in ["train", "all"]:
            # train BLSTM-2DCNN
            train_blstm_2dcnn(argv, blstm_2dcnn_model, loader)
        if argv.mode in ["test", "all"]:
            # test BLSTM-2DCNN
            checkpoint = torch.load(
                os.path.join(argv.checkpoint_root, argv.checkpoint_name))
            blstm_2dcnn_model.load_state_dict(checkpoint['model_state_dict'])
            test_blstm_2dcnn(blstm_2dcnn_model, loader, only=True)

    print()


if __name__ == "__main__":
    # main(parse_arguments(sys.argv[1:]))
    main(
        parse_arguments("all blstm "
                        "--gpu_id 1 "
                        "--data_root ./src_zzc/data "
                        "--fold 10 "
                        "--seed 1586540426 "
                        # "--use_checkpoint "
                        "--checkpoint_name checkpoint "
                        # "--checkpoint_ver 10 "
                        "--epoch 30 "
                        "--batch_size 50 "
                        "--optimizer Adam "
                        "--activate_func ReLU "
                        "--lr 0.01 "
                        # "--is_glove "
                        "--desired_len_percent 0.5 "
                        "--emb_drop_r 0.5 "
                        "--emb_size 200 "
                        "--lstm_drop_r 0.2 "
                        "--lstm_n_layer 1 "
                        "--lstm_hidden_sz 100 "
                        "--cnn_n_kernel 50 "
                        "--cnn_kernel_sz 3 "
                        "--cnn_pool_kernel_sz 2 ".split()))
