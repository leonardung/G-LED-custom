#!/bin/bash

#SBATCH -o /home/l1148900/dns_project/G-LED-custom/output/maple_gpu_vit_job.out
#SBATCH -e /home/l1148900/dns_project/G-LED-custom/output/maple_gpu_vit_job.err
#SBATCH -p maple
#SBATCH -c 72
#SBATCH --mem=64G
#SBATCH --account=upstream
#SBATCH --gres=gpu:1

source /home/l1148900/.bashrc
conda activate base
conda activate py311
python main_seq_train.py
# python main_seq_test.py
