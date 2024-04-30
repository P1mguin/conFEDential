#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --ntasks=1
#SBATCH --output hpc_runs/slurm_outputs/performance_simulation/mnist/models/lr/slurm-%j.out
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 8


# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
source venv/bin/activate

srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/logistic_regression/fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-LOGISTICREGRESSION-FEDADAM
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/logistic_regression/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-LOGISTICREGRESSION-FEDAVG
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/logistic_regression/fed_nag.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-LOGISTICREGRESSION-FEDNAG
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/logistic_regression/fed_nl.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-LOGISTICREGRESSION-FEDNL
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/logistic_regression/scaffold.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-LOGISTICREGRESSION-SCAFFOLD
