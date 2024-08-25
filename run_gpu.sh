#SBATCH --job-name=train_GLM        # job name (default is the name of this file)
#SBATCH --output=slurm/log.%x.job_%j  # file name for stdout/stderr (%x will be replaced with the job name, %j with the jobid)
#SBATCH --time=12:00:00          # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --partition=students        # put the job into the gpu partition
#SBATCH --gres=gpu:1            # number of GPUs per node (gres=gpu:N)

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate GLM
cd 