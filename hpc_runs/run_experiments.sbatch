#!/bin/bash

# Submit the sbatch jobs on exclusive nodes so that the ray cluster is not shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1

srun --nodes=1 --ntasks=1 hpc_runs/cifar10/resnet18.sbatch &
srun --nodes=1 --ntasks=1 hpc_runs/cifar100/resnet34.sbatch &
srun --nodes=1 --ntasks=1 hpc_runs/purchase/fcn.sbatch &
srun --nodes=1 --ntasks=1 hpc_runs/purchase/logistic_regression.sbatch &
srun --nodes=1 --ntasks=1 hpc_runs/texas/fcn.sbatch &
srun --nodes=1 --ntasks=1 hpc_runs/texas/logistic_regression.sbatch &
wait
