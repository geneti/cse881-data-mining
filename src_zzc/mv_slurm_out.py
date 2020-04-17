import os

res_root = "./src_zzc/res_ensemble"
slurm_out = "./slurm_output"

res_root_d = [os.path.join(res_root, f) for f in os.listdir(res_root)]
slurm_out_f = [os.path.join(slurm_out, f) for f in os.listdir(slurm_out)]
slurm_out_f.sort()

for slurm_out_f_path in slurm_out_f:
    sd, ext = os.path.splitext(os.path.basename(slurm_out_f_path))
    for res_d in res_root_d:
        if sd in res_d:
            new_path = os.path.join(res_d, os.path.basename(slurm_out_f_path))
            print(f'mv: {slurm_out_f_path} to {new_path}')
            os.rename(slurm_out_f_path, new_path)

print()