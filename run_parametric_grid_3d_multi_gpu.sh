#!/bin/bash

#SBATCH -o /home/l1148900/dns_project/G-LED-custom/output/vit_3d_parametric/%A_%a.out
#SBATCH -e /home/l1148900/dns_project/G-LED-custom/output/vit_3d_parametric/%A_%a.err

#SBATCH --array=0-23%10 
#SBATCH -p csevolta
#SBATCH -c 4
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --account=upstream

source /home/l1148900/.bashrc
conda activate gled

fps=(2 4)
num_timesteps=(4 8 12)
patch_dim=(8,2,4 16,4,8)
heads=(16 32)

# total lengths
f_len=${#fps[@]}
t_len=${#num_timesteps[@]}
pd_len=${#patch_dim[@]}
h_len=${#heads[@]}

# pick index
i=$SLURM_ARRAY_TASK_ID

# compute indices
f_idx=$(( i / (t_len * pd_len * h_len) ))
r=$(( i % (t_len * pd_len * h_len) ))
t_idx=$(( r / (pd_len * h_len) ))
r=$(( r % (pd_len * h_len) ))
pd_idx=$(( r / h_len ))
h_idx=$(( r % h_len ))

# assign values
f=${fps[$f_idx]}
t=${num_timesteps[$t_idx]}
pd=${patch_dim[$pd_idx]}
h=${heads[$h_idx]}

echo "Task $i: fps=$f num_timesteps=$t patch_dim=$pd heads=$h on GPU $CUDA_VISIBLE_DEVICES"

python parametric_vit_main_seq_train.py \
  --frame_patch_size $f \
  --heads $h \
  --dim_head 512 \
  --num_timesteps $t \
  --patch_dim $pd

# python parametric_vit_main_seq_train.py \
#   --frame_patch_size 2 \
#   --heads 16 \
#   --dim_head 512 \
#   --num_timesteps 12 \
#   --patch_dim 8,8,8