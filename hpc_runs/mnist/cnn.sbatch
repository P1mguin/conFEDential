#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --output hpc_runs/mnist/outputs/cnn/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/main.py --yaml-file examples/mnist/cnn/fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --capturing --run-name MNIST-CNN-FEDADAM
srun python src/main.py --yaml-file examples/mnist/cnn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --capturing --run-name MNIST-CNN-FEDAVG
srun python src/main.py --yaml-file examples/mnist/cnn/fed_nag.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --capturing --run-name MNIST-CNN-FEDNAG
