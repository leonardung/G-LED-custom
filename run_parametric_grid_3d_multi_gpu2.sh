#!/bin/bash

#SBATCH -o /home/l1148900/dns_project/G-LED-custom/output/vit_3d_parametric/%A_%a.out
#SBATCH -e /home/l1148900/dns_project/G-LED-custom/output/vit_3d_parametric/%A_%a.err

#SBATCH --array=0-36%10 
#SBATCH -p csevolta
#SBATCH -c 4
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --account=upstream

source /home/l1148900/.bashrc
conda activate gled

dim_head=(512 768)
heads=(32 48)
dim=(1024 1536 2048)
depth=(6 8 10)

# lengths
dh_len=${#dim_head[@]}
h_len=${#heads[@]}
d_len=${#dim[@]}
depth_len=${#depth[@]}

# pick the SLURM array index
i=${SLURM_ARRAY_TASK_ID}

# compute sub-indices
dh_idx=$(( i / (h_len * d_len * depth_len) ))
r=$(( i % (h_len * d_len * depth_len) ))
h_idx=$(( r / (d_len * depth_len) ))
r=$(( r % (d_len * depth_len) ))
d_idx=$(( r / depth_len))
dep_idx=$(( r % depth_len ))

# assign values
dh=${dim_head[$dh_idx]}
h=${heads[$h_idx]}
d=${dim[$d_idx]}
dep=${depth[$dep_idx]}

echo "Task $i: frame_patch_size=4 num_timesteps=8 patch_dim=16,4,8 \
dim_head=$dh heads=$h dim=$d depth=$dep on GPU $CUDA_VISIBLE_DEVICES"

# launch training
python parametric_vit_main_seq_train.py \
  --frame_patch_size 4 \
  --num_timesteps 8 \
  --patch_dim 16,4,8 \
  --dim_head $dh \
  --heads $h \
  --dim $d \
  --depth $dep

# python parametric_vit_main_seq_train.py \
#   --frame_patch_size 4 \
#   --num_timesteps 8 \
#   --patch_dim 16,4,8 \
#   --dim_head 768 \
#   --heads 48 \
#   --dim 1536 \
#   --depth 8