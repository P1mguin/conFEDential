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
wandb sync --clean

sbatch hpc_runs/run_experiments.sbatch