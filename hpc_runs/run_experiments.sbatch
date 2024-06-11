#!/bin/bash

# Submit the sbatch jobs on exclusive nodes so that the ray cluster is not shared
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --output hpc_runs/outputs/slurm-%j.out

sbatch --nodes=1 --ntasks=1 hpc_runs/cifar10/resnet18.sbatch &
sbatch --nodes=1 --ntasks=1 hpc_runs/cifar100/resnet34.sbatch &
#sbatch --nodes=1 --ntasks=1 hpc_runs/purchase/fcn.sbatch &
#sbatch --nodes=1 --ntasks=1 hpc_runs/purchase/logistic_regression.sbatch &
#sbatch --nodes=1 --ntasks=1 hpc_runs/texas/fcn.sbatch &
#sbatch --nodes=1 --ntasks=1 hpc_runs/texas/logistic_regression.sbatch &
wait
