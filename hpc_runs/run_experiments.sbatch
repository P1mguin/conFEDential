#!/bin/bash

# Submit the sbatch jobs on exclusive nodes so that the ray cluster is not shared
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --output hpc_runs/outputs/slurm-%j.out

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
wandb sync --clean

sbatch hpc_runs/cifar10/resnet18.sbatch &
sbatch hpc_runs/cifar100/resnet34.sbatch &
sbatch hpc_runs/purchase/fcn.sbatch &
sbatch hpc_runs/purchase/logistic_regression.sbatch &
sbatch hpc_runs/texas/fcn.sbatch &
sbatch hpc_runs/texas/logistic_regression.sbatch &
wait
