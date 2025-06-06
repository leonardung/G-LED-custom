#!/bin/bash

#SBATCH -o /home/l1148900/dns_project/G-LED-custom/output/gpu_job.out
#SBATCH -e /home/l1148900/dns_project/G-LED-custom/output/gpu_job.err
#SBATCH -p mvpdev
#SBATCH --account=upstream
#SBATCH --gres=gpu:1

source /home/l1148900/.bashrc
conda activate gled
python main_seq_train.py
