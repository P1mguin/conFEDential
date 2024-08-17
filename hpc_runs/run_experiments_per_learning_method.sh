#!/bin/bash

scripts=(
#    "hpc_runs/texas/logistic_regression/fed_adam.sbatch"            # Not run simulation yet
    "hpc_runs/texas/logistic_regression/fed_avg.sbatch"            # Only attack simulation left
    "hpc_runs/texas/logistic_regression/fed_nag.sbatch"            # Only attack simulation left
    "hpc_runs/texas/logistic_regression/fed_nl.sbatch"             # Only attack simulation left
#    "hpc_runs/texas/fcn/fed_adam.sbatch"                            # Not run simulation yet
    "hpc_runs/texas/fcn/fed_avg.sbatch"                            # Only attack simulation left
    "hpc_runs/texas/fcn/fed_nag.sbatch"                            # Only attack simulation left
#    "hpc_runs/purchase/logistic_regression/fed_adam.sbatch"         # Not run simulation yet
    "hpc_runs/purchase/logistic_regression/fed_avg.sbatch"         # Only attack simulation left
    "hpc_runs/purchase/logistic_regression/fed_nag.sbatch"         # Only attack simulation left
    "hpc_runs/purchase/logistic_regression/fed_nl.sbatch"          # Only attack simulation left
#    "hpc_runs/purchase/fcn/fed_adam.sbatch"                         # Not run simulation yet
#    "hpc_runs/purchase/fcn/fed_avg.sbatch"                         # Only attack simulation left
    "hpc_runs/purchase/fcn/fed_nag.sbatch"                          # Not run simulation yet
#    "hpc_runs/cifar10/resnet18/fed_adam.sbatch"                    # Only attack simulation left
    "hpc_runs/cifar10/resnet18/fed_avg.sbatch"                      # Not run simulation yet
    "hpc_runs/cifar10/resnet18/fed_nag.sbatch"                      # Not run simulation yet
#    "hpc_runs/cifar100/resnet34/fed_adam.sbatch"                   # Only attack simulation left
    "hpc_runs/cifar100/resnet34/fed_avg.sbatch"                     # Not run simulation yet
    "hpc_runs/cifar100/resnet34/fed_nag.sbatch"                     # Not run simulation yet
)

sbatch --nodelist ctit082 hpc_runs/run_experiments.sbatch "${scripts[@]}"
