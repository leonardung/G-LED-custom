#!/bin/bash

#SBATCH -o /home/l1148900/dns_project/G-LED-custom/output/vit64x32x32.out
#SBATCH -e /home/l1148900/dns_project/G-LED-custom/output/vit64x32x32.err
#SBATCH -p mvpdev
#SBATCH -c 8
#SBATCH --mem=32000M
#SBATCH --account=upstream
#SBATCH --gres=gpu:1

source /home/l1148900/.bashrc
conda activate gled
python main_seq_train.py
