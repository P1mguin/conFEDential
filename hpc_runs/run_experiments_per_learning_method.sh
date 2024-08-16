#!/bin/bash

scripts=(
#    "hpc_runs/purchase/logistic_regression/fed_adam.sbatch"
#    "hpc_runs/purchase/logistic_regression/fed_avg.sbatch"
#    "hpc_runs/purchase/logistic_regression/fed_nag.sbatch"
#    "hpc_runs/purchase/logistic_regression/fed_nl.sbatch"
#    "hpc_runs/texas/logistic_regression/fed_adam.sbatch"
#    "hpc_runs/texas/logistic_regression/fed_avg.sbatch"
#    "hpc_runs/texas/logistic_regression/fed_nag.sbatch"
#    "hpc_runs/texas/logistic_regression/fed_nl.sbatch"
#    "hpc_runs/purchase/fcn/fed_adam.sbatch"
    "hpc_runs/purchase/fcn/fed_avg.sbatch"
    "hpc_runs/purchase/fcn/fed_nag.sbatch"
#    "hpc_runs/texas/fcn/fed_adam.sbatch"
    "hpc_runs/texas/fcn/fed_avg.sbatch"
    "hpc_runs/texas/fcn/fed_nag.sbatch"
#    "hpc_runs/cifar10/resnet18/fed_adam.sbatch"
    "hpc_runs/cifar10/resnet18/fed_avg.sbatch"
    "hpc_runs/cifar10/resnet18/fed_nag.sbatch"
#    "hpc_runs/cifar100/resnet34/fed_adam.sbatch"
    "hpc_runs/cifar100/resnet34/fed_avg.sbatch"
    "hpc_runs/cifar100/resnet34/fed_nag.sbatch"
)

sbatch --nodelist caserta hpc_runs/run_experiments.sbatch "${scripts[@]}"
