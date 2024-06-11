#!/bin/bash
# Show node in output file
hostname

cd /home/s2240084/conFEDential
ray start --head --num-cpus 2 --num-gpus 1 --temp-dir /local/ray --memory 17179869184
python src/main.py --yaml-file examples/cifar10/resnet18/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR10-RESNET18-FEDADAM
python src/main.py --yaml-file examples/cifar10/resnet18/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR10-RESNET18-FEDAVG
python src/main.py --yaml-file examples/cifar10/resnet18/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name CIFAR10-RESNET18-FEDNAG
ray stop