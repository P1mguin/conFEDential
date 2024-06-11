#!/bin/bash
# Show node in output file
hostname

cd /home/s2240084/conFEDential
ray start --head --num-cpus 2 --num-gpus 1 --temp-dir /local/ray --memory 17179869184
python src/main.py --yaml-file examples/texas/logistic_regression/fed_adam.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDADAM
python src/main.py --yaml-file examples/texas/logistic_regression/fed_avg.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDAVG
python src/main.py --yaml-file examples/texas/logistic_regression/fed_nag.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDNAG
python src/main.py --yaml-file examples/texas/logistic_regression/fed_nl.yaml --ray --memory 16 --num-cpus 2 --num-gpus 1 --clients 1 --run-name TEXAS-LOG_RES-FEDNL
ray stop
