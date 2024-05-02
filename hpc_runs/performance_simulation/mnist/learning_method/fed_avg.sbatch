#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --output hpc_runs/slurm_outputs/performance_simulation/mnist/learning_method/fed_avg/slurm-%j.out

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/2nn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-2NN-FEDAVG
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/cnn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-CNN-FEDAVG
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/logistic_regression/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-LOGISTICREGRESSION-FEDAVG
