# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/EWC.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/SketchEWC.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/MAS.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/SketchMAS.sh
sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/SCP.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/SketchSCP.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/rotated_EWC.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/rotated_SketchEWC.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/rotated_MAS.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/rotated_SketchMAS.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/rotated_SCP.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/rotated_SketchSCP.sh
# sbatch -o logs/slurm-%A_%a.out --array=1234,2345,3456,4567,5678 bash/SketchSCP_bucket.sh

