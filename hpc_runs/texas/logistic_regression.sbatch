#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=TEXAS-LogRes
#SBATCH --output hpc_runs/texas/outputs/logistic_regression/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

cd /home/s2240084/conFEDential
ray stop
ray start --head --num-cpus 2 --num-gpus 1 --temp-dir /local/ray --memory 36507222016
#srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_adam.yaml --ray --memory 32 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDADAM
#srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_avg.yaml --ray --memory 32 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDAVG
#srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_nag.yaml --ray --memory 32 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDNAG
srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_nl.yaml --ray --memory 32 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDNL
ray stop
