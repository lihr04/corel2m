sbatch -o logs/slurm-%A_%a.out --array=0-4 bash/SketchEWC_10bucket.sh
sbatch -o logs/slurm-%A_%a.out --array=0-4 bash/SketchEWC_bucket.sh
sbatch -o logs/slurm-%A_%a.out --array=0-4 bash/SketchMAS_bucket.sh

