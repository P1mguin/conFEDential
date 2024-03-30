#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=ALL

# Show node in output file
hostname

# Load relevant modules
module load nvidia/nvhpc/23.3
module load python/3.10.7
source venv/bin/activate

srun -n 1 -c 16 --gres=gpu:1 --exclusive python src/batch_run_simulation.py --yaml-file batch_examples/mnist/cnn/fed_avg.yaml --num-gpus 0.125 --num-cpus 2 &
srun -n 1 -c 16 --gres=gpu:1 --exclusive python src/batch_run_simulation.py --yaml-file batch_examples/mnist/logistic_regression/fed_avg.yaml --num-gpus 0.125 --num-cpus 2 &
wait
