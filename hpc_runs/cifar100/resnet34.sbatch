#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --output hpc_runs/cifar100/outputs/resnet34/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/main.py --yaml-file examples/cifar100/resnet34/fed_adam.yaml --memory 16 --clients 1 --run-name CIFAR100-RESNET34-FEDADAM
srun python src/main.py --yaml-file examples/cifar100/resnet34/fed_avg.yaml --memory 16 --clients 1 --run-name CIFAR100-RESNET34-FEDAVG
srun python src/main.py --yaml-file examples/cifar100/resnet34/fed_nag.yaml --memory 16 --clients 1 --run-name CIFAR100-RESNET34-FEDNAG
