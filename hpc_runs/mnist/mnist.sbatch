#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --output hpc_runs/slurm_outputs/mnist/slurm-%j.out
#SBATCH --cpus-per-task 8


# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3

python src/batch_run_simulation.py --yaml-file batch_examples/mnist/cnn/fed_avg.yaml --num-gpus 0 --num-cpus 8
#python src/batch_run_simulation.py --yaml-file batch_examples/mnist/logistic_regression/fed_avg.yaml --num-gpus 0.125 --num-cpus 8

wandb sync --sync-all