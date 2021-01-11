#!/bin/bash
#SBATCH --job-name=EWC
#SBATCH --time=1:0:0
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=hli143@jhu.edu

ml anaconda
ml cuda/10.1
conda activate torch
mkdir -p logs
python grid_search.py -r EWC --id $SLURM_ARRAY_TASK_ID -p -1 0 1 > logs/EWC_$SLURM_ARRAY_TASK_ID
echo "Finished with job $SLURM_JOBID task $SLURM_ARRAY_TASK_ID"
