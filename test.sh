#!/bin/bash
python test.py

# run this script with the following if on slurm:
# srun --partition=gpunodes -c 1 --mem=2G --gres=gpu:1 test.sh