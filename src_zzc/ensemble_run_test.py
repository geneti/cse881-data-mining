import os
import re
import sys
import subprocess

gpu = 0
is_override = False
train_val_path = "./src_zzc/data_ensemble/Training_val.txt"
test_path = "./src_zzc/data/Test.txt"
res_root = "/mnt/gs18/scratch/users/chenzho7/cse881/res_ensemble"

if is_override:
    file_list = [os.path.join(res_root, f,) for f in os.listdir(res_root)] # yapf: disable
    file_list = [f for f in file_list if os.path.isdir(f)]
    file_list = [os.path.join(f, fi) for f in file_list for fi in os.listdir(f)] # yapf: disable
    file_list = [f for f in file_list if os.path.splitext(f)[1] in ['.txt', '.png']] # yapf: disable
    for f in sorted(file_list):
        print(f"RM: {f}")
        os.remove(f)

model_list = dict()
file_list = [os.path.join(res_root, f,) for f in os.listdir(res_root)] # yapf: disable
file_list = [f for f in file_list if os.path.isdir(f)]
file_list = [os.path.join(f, fi) for f in file_list for fi in os.listdir(f)] # yapf: disable
file_list = [f for f in file_list if os.path.splitext(f)[1] == '.pkl'] # yapf: disable

pattern = re.compile(
    f"{res_root}/(?P<model>.*?)_sd(?P<sd>\d*)/chkpt_epoch(?P<epoch>\d*)_acc.*?\.pkl"
)
res = pattern.findall(str(file_list))
for model, sd, ep in res:
    if model not in model_list:
        model_list[model] = []
    model_list[model].append((int(sd), int(ep)))

for key in model_list.keys():
    model_list[key].sort(key=lambda x: x[0])

# model_list = {
#     'blstm': [(1587083616, 15), (1587086570, 17), (1587088177, 11),
#                 (1587089018, 14)]
# }

model_cnt = sum([len(m) for m in model_list.values()])
model_distro_cnt = {key: len(val) for key, val in model_list.items()}
cnt = 0
for model, selected in model_list.items():
    for seed, epoch in selected:
        cnt += 1
        print("#"*150)
        output = os.path.join(res_root, f"{model}_sd{seed}",
                              f"Training_val_pred_epoch{epoch}.txt")

        print(f"[{cnt}/{model_cnt}] Testing: {output}\n")
        sys.stdout.flush()
        if is_override or not os.path.exists(output):
            command = f"./venv/bin/python ./src_zzc/main.py {model} --gpu_id {gpu} --mode test --seed {seed} --checkpoint_ver {epoch} --data_path {train_val_path} --res_root {res_root} --output {output}"
            subprocess.run(command.split())

        output = os.path.join(res_root, f"{model}_sd{seed}",
                              f"test_epoch{epoch}.txt")
        print(f"[{cnt}/{model_cnt}] Testing: {output}\n")
        sys.stdout.flush()
        if is_override or not os.path.exists(output):
            command = f"./venv/bin/python ./src_zzc/main.py {model} --gpu_id {gpu} --mode test --seed {seed} --checkpoint_ver {epoch} --data_path {test_path} --res_root {res_root} --output {output}"
            subprocess.run(command.split())

print()