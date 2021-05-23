#!/bin/bash
#SBATCH --job-name=SketchSCP
#SBATCH --time=5:0:0
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=hli143@jhu.edu

ml anaconda
ml cuda/10.1
conda activate torch
mkdir -p logs
python grid_search.py -r SketchSCP --id $SLURM_ARRAY_TASK_ID -p 4 5 6 7 8 -s 50 -b 50 > logs/SketchSCP_$SLURM_ARRAY_TASK_ID
echo "Finished with job $SLURM_JOBID task $SLURM_ARRAY_TASK_ID"
