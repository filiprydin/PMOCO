#!/bin/bash
#SBATCH -A NAISS2024-22-1251 -p alvis
#SBATCH --job-name=PSL-VRP
#SBATCH --time=168:00:00
#SBATCH --gpus-per-node=A40:1

# Load required modules
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
module load einops/0.7.0-GCCcore-12.3.0

# Run the Python script
python MOTSP/test_motsp_cluster.py
