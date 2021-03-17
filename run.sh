sbatch -o logs/slurm-%A_%a.out --array=0-4 bash/SketchEWC.sh
sbatch -o logs/slurm-%A_%a.out --array=0-4 bash/SketchMAS.sh

