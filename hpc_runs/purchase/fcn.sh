#!/bin/bash
# Show node in output file
hostname

cd /home/s2240084/conFEDential
ray stop
ray start --head --num-cpus 2 --num-gpus 1 --temp-dir /local/ray --memory 17179869184
python src/main.py --yaml-file examples/purchase/fcn/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDADAM
python src/main.py --yaml-file examples/purchase/fcn/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDAVG
python src/main.py --yaml-file examples/purchase/fcn/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name PURCHASE-FCN-FEDNAG
ray stop
