#!/bin/bash

#SBATCH --job-name temp
#SBATCH --output ./temp.log
#SBATCH --account=gliang
#SBATCH --partition=gpuq   
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16                 
#SBATCH --mail-type ALL
#SBATCH --nodelist=SatmresGPU02                # Specify the GPU node
#SBATCH --mail-user gliang@tamusa.edu
#SBATCH --time=23:59:59

bash
echo "Starting job"
date

# Check the GPU status
nvidia-smi

# Activate the conda environment
source /home/gliang/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# Run the script
cd /home/gliang/projects/BetaRisk
python betarisk_train.py
