import os
import numpy as np
import scipy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import utils

train_val_path = "./src_zzc/data_ensemble/Training_val.txt"
test_path = "./src_zzc/data/Test.txt"
res_root = "./src_zzc/res_ensemble"


file_list = [os.path.join(res_root, f,) for f in os.listdir(res_root)] # yapf: disable
file_list = [f for f in file_list if os.path.isdir(f)]
file_list = [os.path.join(f, fi) for f in file_list for fi in os.listdir(f)] # yapf: disable
file_list = [f for f in file_list if os.path.splitext(f)[1] == '.txt'] # yapf: disable


val_pred_all = []
test_pred_all = []

with open("./src_zzc/data_ensemble/Training_val_Label.txt", 'r') as f:
    val_gts = [int(line) for line in f]

nb_cnt = 0
for file_path in file_list:
    if "nb" in file_path:
        continue

    if "Training_val_pred_epoch" in file_path:
        with open(file_path, 'r') as f:
            val_preds = [int(line) for line in f]
            val_pred_all.append(val_preds)
            acc = accuracy_score(val_gts, val_preds)
            print(f"acc: {acc*100:.2f} {file_path}")
    else:
        with open(file_path, 'r') as f:
            test_pred_all.append([int(line) for line in f])

val_pred_all = np.array(val_pred_all)
val_gts = np.array(val_gts)
test_pred_all = np.array(test_pred_all)

assert val_pred_all.shape[1] == val_gts.shape[0]
assert val_pred_all.shape[0] == test_pred_all.shape[0]

val_pred_max_freq, val_pred_max_cnt = scipy.stats.mode(val_pred_all)
target_names = [str(l) for l in range(1, 21)]
print(
    classification_report(val_gts,
                          val_pred_max_freq.T,
                          target_names=target_names))
out_path = os.path.join(res_root, 'ensemble_val.png')
utils.plot_confusion_matrix(target_names, val_gts, val_pred_max_freq.T,
                            out_path)

test_pred_max_freq, test_pred_max_cnt = scipy.stats.mode(test_pred_all)
test_out_path = os.path.join(res_root, 'ensemble_test.txt')
utils.save_preds(test_out_path, test_pred_max_freq.squeeze())
print()