#!/bin/bash

#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=PURCHASE-FCN
#SBATCH --output hpc_runs/purchase/outputs/fcn/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

cd /home/s2240084/conFEDential
ray stop
ray start --head --num-cpus 2 --num-gpus 1 --temp-dir /local/ray --memory 17179869184
srun python src/main.py --yaml-file examples/purchase/fcn/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDADAM
srun python src/main.py --yaml-file examples/purchase/fcn/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDAVG
srun python src/main.py --yaml-file examples/purchase/fcn/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDNAG
ray stop
