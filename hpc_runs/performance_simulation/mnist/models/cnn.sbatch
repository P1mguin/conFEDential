#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 8
#SBATCH --output hpc_runs/slurm_outputs/performance_simulation/mnist/models/cnn/slurm-%j.out


# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/cnn/fed_adam.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-CNN-FEDADAM
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/cnn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-CNN-FEDAVG
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/cnn/fed_nag.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-CNN-FEDNAG
srun python src/simulations/performance/batch_run_simulation.py --yaml-file examples/performance_simulation/batch_experiments/mnist/cnn/scaffold.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --no-capturing --run-name MINST-CNN-SCAFFOLD
