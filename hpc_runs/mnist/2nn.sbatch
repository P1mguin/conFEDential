#!/bin/bash
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --output hpc_runs/slurm_outputs/mnist/slurm-%j.out
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 8


# Show node in output file
hostname

module load python/3.10.7
module load nvidia/nvhpc/23.3
source venv/bin/activate

srun python src/batch_run_simulation.py --yaml-file batch_examples/mnist/2nn/fed_avg.yaml --num-gpus 0.125 --num-cpus 8