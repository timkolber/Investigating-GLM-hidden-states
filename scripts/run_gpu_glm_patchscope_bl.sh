#!/usr/bin/env bash
#SBATCH --job-name=glm        # job name (default is the name of this file)
#SBATCH --output=slurm/log.%x.job_%j  # file name for stdout/stderr (%x will be replaced with the job name, %j with the jobid)
#SBATCH --time=0-48:00:00          # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --partition=students        # put the job into the gpu partition
#SBATCH --gres=gpu:mem11g:1            # number of GPUs per node (gres=gpu:N)
#SBATCH --cpus-per-task=4            # number of CPUs per task
#SBATCH --mail-user=kolber@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL


srun /usr/bin/env /home/students/kolber/miniconda3/envs/GLM/bin/python /home/students/kolber/Investigating-GLM-hidden-states/interpretability/patchscopes/code/GLM_patchscopes_bl.py