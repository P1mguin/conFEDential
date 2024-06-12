#!/bin/bash

# Submit the sbatch jobs on exclusive nodes so that the ray cluster is not shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

srun --nodes=1 --ntasks=1 --output hpc_runs/cifar10/outputs/resnet18/slurm-%j.out hpc_runs/cifar10/resnet18.sh &
srun --nodes=1 --ntasks=1 --output hpc_runs/cifar100/outputs/resnet34/slurm-%j.out hpc_runs/cifar100/resnet34.sh &
srun --nodes=1 --ntasks=1 --output hpc_runs/purchase/outputs/fcn/slurm-%j.out hpc_runs/purchase/fcn.sh &
srun --nodes=1 --ntasks=1 --output hpc_runs/purchase/outputs/logistic_regression/slurm-%j.out hpc_runs/purchase/logistic_regression.sh &
srun --nodes=1 --ntasks=1 --output hpc_runs/texas/outputs/fcn/slurm-%j.out hpc_runs/texas/fcn.sh &
srun --nodes=1 --ntasks=1 --output hpc_runs/texas/outputs/logistic_regression/slurm-%j.out hpc_runs/texas/logistic_regression.sh &
wait
