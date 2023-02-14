#!/bin/bash
#SBATCH --job-name=HCExperiment1

# To get email updates when your job starts, ends, or fails
#SBATCH --mail-user=syip0005@student.monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Replace <project> with your project ID
#SBATCH --account=au31

#SBATCH --time=10:00:00
#SBATCH --ntasks=6
#SBATCH --gres=gpu:T4:1
#SBATCH --partition=gpu

# Edit this section to activate your conda environment
source /scratch/au31/scotty/miniconda/bin/activate
conda activate jupyterlab

# Edit this to point to your repositories location
export REPODIR=/scratch/au31/scotty/hybridcompression

# Run the model

cd /scratch/au31/scotty/hybridcompression

python ${REPODIR}/train.py \
    --gpus=1 \
    --wandb \
    --entity=syip0005 \
    --epochs=2000 \
    --dataset=CERN \
    --model=SCSAE \
    --batch=128 \
    --learning_rate_style=constant \
    --lr=1e-3 \
    --latent_n=48 \
    --batchnorm

EOF