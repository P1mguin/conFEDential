#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --output hpc_runs/slurm_outputs/performance_simulation/mnist/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH -c 8

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/step_experiments/mnist/optimized_fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MNIST-OPTIMIZED-MODELS
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/step_experiments/mnist/optimized_fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MNIST-OPTIMIZED-MODELS
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/step_experiments/mnist/optimized_fed_nag.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MNIST-OPTIMIZED-MODELS
