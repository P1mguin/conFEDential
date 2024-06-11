#!/bin/bash
# Show node in output file
hostname

cd /home/s2240084/conFEDential
ray start --head --num-cpus 2 --num-gpus 1 --temp-dir /local/ray --memory 17179869184
srun python src/main.py --yaml-file examples/texas/fcn/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-FCN-FEDADAM
srun python src/main.py --yaml-file examples/texas/fcn/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-FCN-FEDAVG
srun python src/main.py --yaml-file examples/texas/fcn/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-FCN-FEDNAG
ray stop
