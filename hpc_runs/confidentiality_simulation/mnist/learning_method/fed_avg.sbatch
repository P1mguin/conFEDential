#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=BEGIN,END
#SBATCH --ntasks=1
#SBATCH --output hpc_runs/slurm_outputs/confidentiality_simulation/mnist/learning_method/fed_avg/slurm-%j.out
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 8


# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
source venv/bin/activate

srun python src/simulations/confidentiality/run_simulation.py --yaml-file examples/confidentiality_simulation/single_experiments/mnist/logistic_regression/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --run-name MNIST-LOGISTICREGRESSION-FEDAVG
srun python src/simulations/confidentiality/run_simulation.py --yaml-file examples/confidentiality_simulation/single_experiments/mnist/2nn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --run-name MNIST-2NN-FEDAVG
srun python src/simulations/confidentiality/run_simulation.py --yaml-file examples/confidentiality_simulation/single_experiments/mnist/cnn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8 --no-logging --run-name MNIST-CNN-FEDAVG
