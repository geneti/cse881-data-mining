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
from DAN import DAN
from BLSTM import BLSTM
from BLSTM_2DCNN import BLSTM_2DCNN

MODE = ["train", "test"]
MODEL = ["nb", "dan", "blstm", "blstm_2dcnn"]
ACTIVATION_FUNC = ["ReLU", "Tanh"]
OPTIMIZER = ["Adam", "SGD"]

# USE_CUDA = False
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def parse_arguments(argv):
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("model",        default="E2ENCR", type=str.lower,  choices=MODEL, help="which model to use",)
    parser.add_argument("--is_redirected", action="store_true",    help="whether or not output is redirected",)
    parser.add_argument("--is_disk_limited", action="store_true",    help="whether or not to limit chpkt saving",)
    parser.add_argument("--mode",       default="test", type=str.lower, choices=MODE, help="whether to train or test the model.",)
    parser.add_argument("--gpu_id",     default=0, type=int, help="gpu id")
    parser.add_argument("--max_idx",    default=161131, type=int, help="maximum index in the data")
    parser.add_argument("--data_path",  default="./src_zzc/data/Test.txt", type=str, help="path to the data",)
    parser.add_argument("--res_root",   default="./src_zzc/results", type=str, help="root for the results",)
    parser.add_argument("--output",     default=None, type=str, help="path to the data",)
    parser.add_argument("--fold",       default=5, type=int, help="number of fold")
    parser.add_argument("--num_label",  default=20, type=int, help="number of fold")
    parser.add_argument("--use_checkpoint",     action="store_true",    help="whether or not to use checkpoint",)
    parser.add_argument("--checkpoint_name",    default='chkpt',   type=str, help="checkpoint file name", )
    parser.add_argument("--checkpoint_ver",     default=1,              type=int, help="checkpoint version")
    parser.add_argument("--seed",       default=int(time.time()), type=int, help="seed used for randomness",)

    # DAN feed forward net
    parser.add_argument("--dan_hidden_sz",      default=100, type=int, help="number of kernel/filter for cnn")
    parser.add_argument("--dan_num_hidden",      default=3,   type=int, help="kernel size for cnn")

    # BLSTM-2DCNN
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
    argv.model_folder = f"{argv.model}_sd{argv.seed}"
    return argv


def valid_model(model, loader, epoch, is_nn):
    out_path = os.path.join(model.argv.res_root, model.argv.model_folder,
                            f'{type(model).__name__}_validation_epoch%d.png')
    if is_nn:
        loss_func = nn.CrossEntropyLoss(FloatTensor(loader.data_weights))
        test_losses = []
        preds_list = []
        gts_list = []
        for batch in loader.get_batch(val=True):
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
        return acc
    else:
        preds = model.test(loader.data_val)
        gts = list(zip(*loader.data_val))[0]

        acc = accuracy_score(gts, preds)
        print(f'\tThe accuracy is %.2f%%' % (acc * 100)) # yapf: disable
        target_names = [str(l) for l in range(1, model.argv.num_label + 1)]
        print(classification_report(gts, preds, target_names=target_names))
        utils.plot_confusion_matrix(model.labels, gts, preds, out_path % epoch)
        return acc


def train_model(model, loader, checkpoint, is_nn):
    print(f"Training {type(model).__name__}")
    argv = model.argv

    chkpt_path = os.path.join(argv.res_root, argv.model_folder,
                              f'{argv.checkpoint_name}_epoch%d_acc%.2f.pkl')
    chkpt_save_path=''
    if is_nn:
        loss_func = nn.CrossEntropyLoss(FloatTensor(loader.data_weights))
        param = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = (optim.Adam(param, lr=argv.lr) if argv.optimizer == "Adam"
                     else optim.SGD(param, lr=argv.lr))

        if argv.use_checkpoint:
            print(f"Training: Loading training state")
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
            for i, batch in enumerate(loader.get_batch(val=False)):
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

                if not argv.is_redirected:
                    print(
                        f"\r\t[{i+1:03d}/{math.ceil(len(loader.data) / argv.batch_size)}] "
                        f"batch: loss : {np.mean(losses):0.2f}, acc: {np.mean(acc):0.2f}% | ",
                        end='')
            print(
                f"\t[{epoch:02d}/{argv.epoch+epoch_offset-1}] "
                f"train: loss : {np.mean(losses):0.2f}, acc: {np.mean(acc):0.2f}% | ",
                end='')
            val_acc = valid_model(model, loader, epoch, is_nn)

            sys.stdout.flush()
            is_save = True
            if argv.is_disk_limited and os.path.exists(chkpt_save_path):
                acc_idx = chkpt_save_path.index('acc')
                ext_idx = chkpt_save_path.index('.pkl')
                prev_acc = float(chkpt_save_path[acc_idx + 3:ext_idx])
                diff = prev_acc - val_acc * 100
                if diff > 0.01:
                    is_save = False
                elif diff < 0.01:
                    os.remove(chkpt_save_path)

            if is_save:
                chkpt_save_path = chkpt_path % (epoch, val_acc * 100)
                torch.save(
                    {
                        'argv': argv,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, chkpt_path % (epoch, val_acc * 100))
    else:
        model.train(loader.data)
        val_acc = valid_model(model, loader, 1, is_nn)
        torch.save(
            {
                'argv': argv,
                'epoch': 1,
                'model_state_dict': model.state_dict(),
            }, chkpt_path % (1, val_acc * 100))


def test_model(model, loader, is_nn):
    print(f"Testing {type(model).__name__}")
    if is_nn:
        preds_list = []
        for batch in loader.get_batch(val=False):
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
        preds = preds[:len(loader.data)] + 1
    else:
        preds = model.test(loader.data)

    if model.argv.output is None:
        out_path = os.path.join(model.argv.res_root, model.argv.model_folder,
                                f'test_epoch{model.argv.checkpoint_ver}.txt')
    else:
        out_path = model.argv.output
    utils.save_preds(out_path, preds)


def create_model(argv, loader):
    if argv.model == "nb":
        model = NB_Model(argv)
    elif argv.model == "dan":
        model = DAN(argv)
    elif argv.model == "blstm":
        model = BLSTM(argv, loader.desired_len)
    elif argv.model == "blstm_2dcnn":
        model = BLSTM_2DCNN(argv, loader.desired_len)

    return model


def main(argv):
    print("Torch GPU set:", argv.gpu_id)
    torch.manual_seed(argv.seed)
    print("Torch Seed was:", argv.seed)
    utils.mkdir(os.path.join(argv.res_root, argv.model_folder))

    loader = DataLoader(argv)
    loader.read_data()
    argv.desired_len = loader.desired_len
    # assert loader.max_idx == argv.max_idx

    is_nn = (argv.model != 'nb')
    model = create_model(argv, loader)

    # load chkpt
    checkpoint = None
    if argv.mode == 'test' or argv.use_checkpoint:

        chkpt_load_path = None
        for file in os.listdir(os.path.join(argv.res_root, argv.model_folder)):
            if f"{argv.checkpoint_name}_epoch{argv.checkpoint_ver}" in file:
                chkpt_load_path = os.path.join(argv.res_root,
                                               argv.model_folder, file)
                break
        if chkpt_load_path is None:
            raise Exception("Can't find checkpoint")

        print(f"\tLoading {chkpt_load_path}")
        checkpoint = torch.load(chkpt_load_path)
        # old argv content that still want to keep
        checkpoint['argv'].output = argv.output
        checkpoint['argv'].mode = argv.mode
        checkpoint['argv'].use_checkpoint = argv.use_checkpoint
        assert checkpoint['epoch'] == argv.checkpoint_ver
        checkpoint['argv'].checkpoint_ver = argv.checkpoint_ver

        argv = checkpoint['argv']
        epoch = checkpoint['epoch'] + 1
        model = create_model(argv, loader)
        loader.desired_len = argv.desired_len
        loader.batch_size = argv.batch_size

    if USE_CUDA and is_nn:
        torch.cuda.set_device(argv.gpu_id)
        model = model.cuda()

    print(f"\n{argv.mode} {type(model).__name__} {'#'*50}")
    if argv.mode == 'test':
        model.load_state_dict(checkpoint['model_state_dict'])
        test_model(model, loader, is_nn)
    else:
        train_model(model, loader, checkpoint, is_nn)

    print()


if __name__ == "__main__":
    # tmp = "nb --mode train --data_path ./src_zzc/data/Training.txt "
    # tmp = "nb --mode test --seed 1587061163 --data_path ./src_zzc/data/Test.txt "
    tmp = "blstm --mode train --is_disk_limited --epoch 20 --fold 5 --batch_size 50 --desired_len_percent 1 --optimizer Adam --lr 0.01 --activate_func ReLU --emb_drop_r 0.5 --emb_size 100 --lstm_drop_r 0 --lstm_n_layer 1 --lstm_hidden_sz 100 --data_path ./src_zzc/data_ensemble/Training.txt --res_root ./src_zzc/res_ensemble "
    # tmp = "blstm --mode train --seed 1587064891 --use_checkpoint --checkpoint_ver 7 --epoch 20 --batch_size 50 --desired_len_percent 1 --optimizer Adam --lr 0.01 --activate_func ReLU --emb_drop_r 0.5 --emb_size 100 --lstm_drop_r 0 --lstm_n_layer 1 --lstm_hidden_sz 100 --data_path ./src_zzc/data/Training.txt "
    # tmp = "blstm --mode test --seed 1587064891 --checkpoint_ver 8 --data_path ./src_zzc/data/Test.txt "

    # print(tmp)
    # main(parse_arguments(tmp.split()))
    print(sys.argv[1:])
    main(parse_arguments(sys.argv[1:]))
