import os
import utils
from LoadData import DataLoader
from argparse import Namespace

data_path = "./src_zzc/data/Training.txt"
dest = "./src_zzc/data_ensemble"

argv = Namespace()
argv.data_path = data_path
argv.batch_size = 50
argv.fold = 5
argv.mode = 'train'
argv.desired_len_percent = 0.5
argv.num_label = 20
argv.seed = 1587064891

loader = DataLoader(argv)
loader.read_data()

train = open(os.path.join(dest, 'Training.txt'), 'w')
train_label = open(os.path.join(dest, 'Training_Label.txt'), 'w')
for gt, msg in loader.data:
    train_label.write(f"{gt}\n")
    train.write(",".join([str(word) for word in msg]))
    train.write("\n")

train.close()
train_label.close()

train = open(os.path.join(dest, 'Training_val.txt'), 'w')
train_label = open(os.path.join(dest, 'Training_val_Label.txt'), 'w')
for gt, msg in loader.data_val:
    train_label.write(f"{gt}\n")
    train.write(",".join([str(word) for word in msg]))
    train.write("\n")

train.close()
train_label.close()

print()
