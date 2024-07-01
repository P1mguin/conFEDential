#!/bin/bash

# Pull the latest changes from the git repository
git pull

# Load the necessary modules
module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node

# Activate the virtual environment
source venv/bin/activate

# Sync all runs with wandb and clean up
wandb sync --sync-all
wandb sync --clean-force

sbatch --exclude=caserta,ctit081,ctit087 hpc_runs/cifar10/resnet18.sbatch
sbatch --exclude=caserta,ctit081,ctit087 hpc_runs/cifar100/resnet34.sbatch
sbatch --exclude=caserta,ctit081,ctit087 hpc_runs/purchase/fcn.sbatch
sbatch --exclude=caserta,ctit081,ctit087 hpc_runs/purchase/logistic_regression.sbatch
sbatch --exclude=caserta,ctit081,ctit087 hpc_runs/texas/fcn.sbatch
sbatch --exclude=caserta,ctit081,ctit087 hpc_runs/texas/logistic_regression.sbatch
