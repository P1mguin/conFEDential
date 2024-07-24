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
default_exclusion_list="caserta,ctit081,ctit082,ctit087"
get_exclusion_list(){
    local user=s2240084
    local exclusion_list="caserta,ctit081,ctit082,ctit087"
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
    if [ -z "$exclusion_list" ]; then
        sbatch $script
    else
        sbatch --exclude=$exclusion_list $script
    fi
}

if [ $# -eq 0 ]; then
  echo "No scripts provided"
  exit 1
fi

for script in "$@"; do
  submit_job $script &
done

wait

while true; do
  sleep 60

  # Check the status of the jobs of s2240084
  user_jobs=$(squeue -u s2240084 -h -o "%A %t %N")

  # Process each job
  while read -r job_id job_status job_node; do
    if [[ "$job_status" == "PD" ]]; then
      # Job is pending, check if the exclusion list has changed
      exclusion_list=$(get_exclusion_list)
      current_exclusion=$(scontrol show job $job_id | grep "ExcNodeList=" | cut -d= -f2)

      if [ "$exclusion_list" != "$current_exclusion" ]; then
        # Re-submit the job with the updated exclusion list
        scancel $job_id
        script=$(scontrol show job $job_id | grep "Command=" | cut -d= -f2)
        submit_job $script &
      fi
    fi

    done <<< "$user_jobs"

    wait
done