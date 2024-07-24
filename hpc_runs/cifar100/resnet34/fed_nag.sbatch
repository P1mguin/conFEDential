#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=CIFAR100-ResNet34-FedNag
#SBATCH --output hpc_runs/cifar100/outputs/resnet34/slurm-%j.out
#SBATCH --mail-type=END

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

cd /home/s2240084/conFEDential
ray stop
ray start --head --num-cpus 4 --temp-dir /local/ray --memory 17179869184
srun python src/main.py --yaml-file examples/cifar100/resnet34/fed_nag.yaml --ray --memory 16 --num-cpus 4 --clients 1 --run-name CIFAR100-RESNET34-FEDNAG --no-capturing
ray stop
