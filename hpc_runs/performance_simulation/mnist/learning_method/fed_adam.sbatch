#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --output hpc_runs/slurm_outputs/performance_simulation/mnist/learning_method/fed_adam/slurm-%j.out
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 8


# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
source venv/bin/activate

srun python src/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/2nn/fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-CNN-FEDADAM
srun python src/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/cnn/fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-2NN-FEDADAM
srun python src/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/logistic_regression/fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-LOGISTICREGRESSION-FEDADAM
