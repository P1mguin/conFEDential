#!/bin/bash
#SBATCH --partition=main
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --output hpc_runs/slurm_outputs/mnist/slurm-%j.out

# Show node in output file
hostname

# Load relevant modules
module load python/3.10.7
module load nvidia/nvhpc/23.3
source venv/bin/activate

#srun -n 1 -c 8 --gres=gpu:1 --exclusive python src/batch_run_simulation.py --yaml-file batch_examples/mnist/cnn/fed_avg.yaml --num-gpus 0.125 --num-cpus 4 &
srun -n 1 -c 8 --gres=gpu:1 --exclusive python src/batch_run_simulation.py --yaml-file batch_examples/mnist/logistic_regression/fed_avg.yaml --num-gpus 0.125 --num-cpus 4 &
wait

wandb sync --sync-all