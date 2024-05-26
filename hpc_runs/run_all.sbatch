#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --output hpc_runs/outputs/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

cd /home/s2240084/conFEDential
ray start --head --num-cpus 2 --num-gpus 1 --temp-dir /local/ray --memory 17179869184

#Purchase FCN
srun python src/main.py --yaml-file examples/purchase/fcn/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDADAM
srun python src/main.py --yaml-file examples/purchase/fcn/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDAVG
srun python src/main.py --yaml-file examples/purchase/fcn/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDNAG

#Purchase Logistic Regression
srun python src/main.py --yaml-file examples/purchase/logistic_regression/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-LOG_RES-FEDADAM
srun python src/main.py --yaml-file examples/purchase/logistic_regression/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-LOG_RES-FEDAVG
srun python src/main.py --yaml-file examples/purchase/logistic_regression/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-LOG_RES-FEDNAG
srun python src/main.py --yaml-file examples/purchase/logistic_regression/fed_nl.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-LOG_RES-FEDNL

#Texas Logistic Regression
srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDADAM
srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDAVG
srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDNAG
srun python src/main.py --yaml-file examples/texas/logistic_regression/fed_nl.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDNL

#CIFAR10 ResNet18
srun python src/main.py --yaml-file examples/cifar10/resnet18/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR10-RESNET18-FEDADAM
srun python src/main.py --yaml-file examples/cifar10/resnet18/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR10-RESNET18-FEDAVG
srun python src/main.py --yaml-file examples/cifar10/resnet18/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR10-RESNET18-FEDNAG

#CIFAR100 ResNet34
srun python src/main.py --yaml-file examples/cifar100/resnet34/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR100-RESNET34-FEDADAM
srun python src/main.py --yaml-file examples/cifar100/resnet34/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR100-RESNET34-FEDAVG
srun python src/main.py --yaml-file examples/cifar100/resnet34/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR100-RESNET34-FEDNAG

ray stop
