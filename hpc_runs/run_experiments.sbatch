#!/bin/bash

#SBATCH --job-name=job_deployer
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=NONE
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --parsable

# Load necessary modules
module load python/3.10.7
module load nvidia/nvhpc/23.3
module load slurm/utils
module load monitor/node

# Activate the virtual environment
source venv/bin/activate

# Function to get the list of nodes used by the user s2240084
get_exclusion_list(){
    local user=s2240084
    local exclusion_list="caserta,ctit080,ctit081,ctit082,ctit086,ctit087,ctit090,ctit091"
    for job in $(squeue -u $user -h -o "%i"); do
        node=$(squeue -j $job -h -o "%N")
        if [ -n "$node" ]; then
            if [ -n "$exclusion_list" ]; then
                exclusion_list="$exclusion_list,$node"
            else
                exclusion_list="$node"
            fi
        fi
    done
    echo "$exclusion_list"
}

# Function to submit a job while excluding certain nodes
submit_job() {
    local script=$1
    local exclusion_list=$(get_exclusion_list)
    local sbatch_output=$(sbatch --exclude=$exclusion_list $script)
    local job_id=$(echo " $sbatch_output" | awk '{print $4}')
    echo "$job_id"
}

if [ $# -eq 0 ]; then
  echo "No scripts provided"
  exit 1
fi

for script in "$@"; do
    # Store how many jobs are running
    running_jobs=$(squeue -u s2240084 -h -t R -o "%i" | wc -l)

    # Submit job with exclusion list
    job_id=$(submit_job $script)

    # While job is pending, if the amount of running jobs decreases, update amount of running jobs and
    # resubmit with new exclusion list
    while true; do
      job_status=$(squeue -j $job_id -h -o "%T")
      if [ "$job_status" == "PENDING" ]; then
        new_running_jobs=$(squeue -u s2240084 -h -t R -o "%i" | wc -l)
        exclusion_list=$(get_exclusion_list)
        if [ $new_running_jobs -lt $running_jobs ]; then
          # Cancel the job id and resubmit with new exclusion list
          scancel $job_id
          running_jobs=$new_running_jobs
          job_id=$(submit_job $script)
        fi
      else
        break
      fi
      sleep 900
    done
done
