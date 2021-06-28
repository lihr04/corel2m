#!/bin/bash
#SBATCH --job-name=SketchSCP_bucket
#SBATCH --time=8:0:0
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=end
#SBATCH --mail-user=hli143@jhu.edu

ml anaconda
ml cuda/10.1
conda activate torch
mkdir -p logs
python grid_search.py -r SketchSCP --id $SLURM_ARRAY_TASK_ID -p 2 -s 10 -b 10 20 30 40 50 --result-filename "bucket_10-50" > logs/SketchSCP_bucket_$SLURM_ARRAY_TASK_ID
echo "Finished with job $SLURM_JOBID task $SLURM_ARRAY_TASK_ID"

