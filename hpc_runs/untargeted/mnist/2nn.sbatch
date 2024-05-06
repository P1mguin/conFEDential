#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --output hpc_runs/untargeted/mnist/outputs/2nn/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/main.py --yaml-file examples/untargeted/mnist/2nn/fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --capturing --run-name MNIST-2NN-FEDADAM
srun python src/main.py --yaml-file examples/untargeted/mnist/2nn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --capturing --run-name MNIST-2NN-FEDAVG
srun python src/main.py --yaml-file examples/untargeted/mnist/2nn/fed_nag.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --capturing --run-name MNIST-2NN-FEDNAG
