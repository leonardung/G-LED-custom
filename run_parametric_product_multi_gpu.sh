#!/bin/bash

#SBATCH -o /home/l1148900/dns_project/G-LED-custom/output/vit_parametric.out
#SBATCH -e /home/l1148900/dns_project/G-LED-custom/output/vit_parametric.err

#SBATCH --array=0-17%10 
#SBATCH -p csevolta
#SBATCH -c 4
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --account=upstream

source /home/l1148900/.bashrc
conda activate gled

fps=(2 4)
heads=(16 32 64)
dims=(128 256 512)

i=$SLURM_ARRAY_TASK_ID
f=${fps[$((i/9))]}
h=${heads[$(((i/3)%3))]}
d=${dims[$((i%3))]}

echo "Task $i: fps=$f heads=$h dim_head=$d on GPU $CUDA_VISIBLE_DEVICES"

python parametric_vit_main_seq_train.py \
  --frame_patch_size $f \
  --heads $h \
  --dim_head $d \
