#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --output hpc_runs/slurm_outputs/performance_simulation/mnist/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G

# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node
source venv/bin/activate

srun python src/simulations/confidentiality/run_simulation.py --yaml-file examples/confidentiality_simulation/single_experiments/mnist/logistic_regression/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --run-name MNIST-LOGISTICREGRESSION-FEDAVG
srun python src/simulations/confidentiality/run_simulation.py --yaml-file examples/confidentiality_simulation/single_experiments/mnist/2nn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --run-name MNIST-2NN-FEDAVG
srun python src/simulations/confidentiality/run_simulation.py --yaml-file examples/confidentiality_simulation/single_experiments/mnist/cnn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --run-name MNIST-CNN-FEDAVG
