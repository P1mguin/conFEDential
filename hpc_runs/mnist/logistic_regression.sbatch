#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --output hpc_runs/mnist/outputs/logistic_regression/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/main.py --yaml-file examples/mnist/logistic_regression/fed_adam.yaml --clients 8 --no-logging --capturing --run-name MNIST-LOGISTICREGRESSION-FEDADAM
srun python src/main.py --yaml-file examples/mnist/logistic_regression/fed_avg.yaml --clients 8 --no-logging --capturing --run-name MNIST-LOGISTICREGRESSION-FEDAVG
srun python src/main.py --yaml-file examples/mnist/logistic_regression/fed_nag.yaml --clients 8 --no-logging --capturing --run-name MNIST-LOGISTICREGRESSION-FEDNAG
