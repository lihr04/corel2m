sbatch -o logs/slurm-%A_%a.out --array=0 bash/SketchEWC_add.sh
sbatch -o logs/slurm-%A_%a.out --array=0 bash/SketchMAS_add.sh

