#!/bin/bash

#SBATCH --mail-type=NONE
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=job-deployer
#SBATCH --output=/dev/null

scripts=(
    "hpc_runs/purchase/logistic_regression/fed_adam.sbatch"
    "hpc_runs/purchase/logistic_regression/fed_avg.sbatch"
    "hpc_runs/purchase/logistic_regression/fed_nag.sbatch"
    "hpc_runs/purchase/logistic_regression/fed_nl.sbatch"
    "hpc_runs/texas/logistic_regression/fed_adam.sbatch"
    "hpc_runs/texas/logistic_regression/fed_avg.sbatch"
    "hpc_runs/texas/logistic_regression/fed_nag.sbatch"
    "hpc_runs/texas/logistic_regression/fed_nl.sbatch"
    "hpc_runs/purchase/fcn/fed_adam.sbatch"
    "hpc_runs/purchase/fcn/fed_avg.sbatch"
    "hpc_runs/purchase/fcn/fed_nag.sbatch"
    "hpc_runs/texas/fcn/fed_adam.sbatch"
    "hpc_runs/texas/fcn/fed_avg.sbatch"
    "hpc_runs/texas/fcn/fed_nag.sbatch"
    "hpc_runs/cifar10/resnet18/fed_adam.sbatch"
    "hpc_runs/cifar10/resnet18/fed_avg.sbatch"
    "hpc_runs/cifar10/resnet18/fed_nag.sbatch"
    "hpc_runs/cifar100/resnet34/fed_adam.sbatch"
    "hpc_runs/cifar100/resnet34/fed_avg.sbatch"
    "hpc_runs/cifar100/resnet34/fed_nag.sbatch"
)

# Function to get the computation nodes on which jobs of a user are already running
function get_exclusion_list() {
    local user=$1
    local exclusion_list="caserta,ctit081,ctit087"
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

# Function to submit a job excluding a set of computation and to get the job id
function submit_job() {
    local script=$1
    local exclusions=$2
    if [ -z "$exclusions" ]; then
      sbatch_output=$(sbatch $script)
    else
      sbatch_output=$(sbatch --exclude=$exclusions $script)
    fi
    job_id=$(echo "$sbatch_output" | awk '{print $4}')
    echo "$job_id"
}

# Function to get the node of a running job
function get_node() {
    local job_id=$1
    node=$(squeue -j $job_id -h -o "%N")
    echo "$node"
}

for script in "${scripts[@]}"; do
    # Submit jobs and build exclusion list
    exclusion_list=$(get_exclusion_list $USER)
    job_id=$(submit_job $script "$exclusion_list")

    # Wait until the job is assigned to a node
    while true; do
      node=$(get_node $job_id)
      if [ -n "$node" ]; then
        break
      fi
      sleep 1
    done

    # Add the node to the exclusion list
    if [ -n "$exclusion_list" ]; then
      exclusion_list="$exclusion_list,$node"
    else
      exclusion_list="$node"
    fi

    # Log the job and node
    echo "Submitted $script as job $job_id on node $node"
done
