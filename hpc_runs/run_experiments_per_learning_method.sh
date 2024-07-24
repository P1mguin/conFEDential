#!/bin/bash

# Load the necessary modules
module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node

# Activate the virtual environment
source venv/bin/activate

# Sync all runs with wandb and clean up
wandb sync --sync-all
wandb sync --clean --clean-old-hours 0

sbatch --nodelist caserta hpc_runs/run_experiments_per_learning_method.sbatch
