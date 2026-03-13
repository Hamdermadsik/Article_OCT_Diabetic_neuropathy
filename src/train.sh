#!/bin/bash
#BSUB -J segmentation_train          # Job name
#BSUB -q gpuv100                     # Queue name
#BSUB -n 4                           # Number of CPU cores
#BSUB -R "span[hosts=1]"             # Run on a single node
#BSUB -R "rusage[mem=16GB]"          # Memory per host
#BSUB -W 00:40                       # Wall time (HH:MM)
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU
#BSUB -o segmentation_train_%J.out    # Standard output
#BSUB -e segmentation_train_%J.err    # Standard error

# --- ENVIRONMENT SETUP ---
# Go to your project root (one level up from src/)
cd /zhome/29/b/146867/Article_OCT/Article_OCT_Diabetic_neuropathy || exit

# Activate your virtual environment

source .venv/bin/activate


# Ensure the root directory and src directory are in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# --- RUN TRAINING ---
# Passing "$@" allows you to override hyperparameters via CLI:
# bsub < src/train.sh hyperparameters.learning_rate=0.005
python src/train.py "$@"
