#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=TEXAS-LogRes-FedAdam
#SBATCH --output hpc_runs/texas/outputs/logistic_regression/slurm-%j.out
#SBATCH --mail-type=END

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

cd /home/s2240084/conFEDential
ray stop
ray start --head --num-cpus 4 --temp-dir /local/ray --memory 36507222016
srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_adam.yaml --ray --memory 32 --num-cpus 4 --clients 1 --run-name TEXAS-LOG_RES-FEDADAM --no-capturing
ray stop
