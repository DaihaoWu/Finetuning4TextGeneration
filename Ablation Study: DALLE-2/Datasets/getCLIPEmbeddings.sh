#!/bin/bash
#SBATCH --job-name=clip_embeddings
#SBATCH --output=%x_%j.out          # Save output with job name and job ID
#SBATCH --error=%x_%j.err           # Save errors with job name and job ID
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --mem=2G                   # Memory per node, adjust as necessary
#SBATCH --gres=gpu:1                # Request 1 GPU, adjust as necessary

# Activate conda environment
source /w/340/michaelyuan/miniconda3/etc/profile.d/conda.sh 
conda activate dalle               # Activate your dalle environment

python CLIPEmbeddings.py