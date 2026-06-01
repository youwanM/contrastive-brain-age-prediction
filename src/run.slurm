#!/bin/bash
#SBATCH --job-name=VAE_H100
#SBATCH --account=gao@h100
#SBATCH --partition=gpu_p6         # H100 Partition
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=logs/AnatCL_%j.out

module purge
module load python/3.10.4
source $WORK/AnatCL/venv/bin/activate

export WANDB_MODE=offline
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

srun python -u launcher.py exp/supcon_adam_kernel.yaml
