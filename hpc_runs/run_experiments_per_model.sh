#!/bin/bash

scripts=(
    "hpc_runs/purchase/logistic_regression.sbatch"
    "hpc_runs/texas/logistic_regression.sbatch"
    "hpc_runs/purchase/fcn.sbatch"
    "hpc_runs/texas/fcn.sbatch"
    "hpc_runs/cifar10/resnet18.sbatch"
    "hpc_runs/cifar100/resnet34.sbatch"
)

sbatch --nodelist caserta hpc_runs/run_experiments.sbatch "${scripts[@]}"
