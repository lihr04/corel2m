sbatch -o logs/slurm-%A_%a.out --array=0 bash/SCP.sh
sbatch -o logs/slurm-%A_%a.out --array=0 bash/SketchSCP.sh

