#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=CIFAR10-ResNet18
#SBATCH --output hpc_runs/cifar10/outputs/resnet18/slurm-%j.out

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
#srun python src/main.py --yaml-file examples/cifar10/resnet18/fed_adam.yaml --ray --memory 16 --num-cpus 4 --clients 1 --run-name CIFAR10-RESNET18-FEDADAM
#srun python src/main.py --yaml-file examples/cifar10/resnet18/fed_avg.yaml --ray --memory 16 --num-cpus 4 --clients 1 --run-name CIFAR10-RESNET18-FEDAVG
srun python src/main.py --yaml-file examples/cifar10/resnet18/fed_nag.yaml --ray --memory 16 --num-cpus 4 --clients 1 --run-name CIFAR10-RESNET18-FEDNAG
ray stop