#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

#SBATCH -n 4
#SBATCH --mem=20G
#SBATCH -t 12:00:00

module load cuda

python train.py

