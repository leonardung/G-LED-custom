#!/bin/bash

#SBATCH -o /home/l1148900/dns_project/G-LED-custom/output/vit_parametric.out
#SBATCH -e /home/l1148900/dns_project/G-LED-custom/output/vit_parametric.err

#SBATCH --array=0-2%3     
#SBATCH -p csevolta
#SBATCH -c 4
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --account=upstream

source /home/l1148900/.bashrc
conda activate gled

# define exactly your three cases
fps_list=(4    4     4)
heads_list=(16  16    32)
dims_list=(256  512   512)

i=$SLURM_ARRAY_TASK_ID
f=${fps_list[$i]}
h=${heads_list[$i]}
d=${dims_list[$i]}

echo "Task $i: fps=$f heads=$h dim_head=$d on GPU $CUDA_VISIBLE_DEVICES"

python parametric_vit_main_seq_train.py \
  --frame_patch_size $f \
  --heads $h \
  --dim_head $d \
