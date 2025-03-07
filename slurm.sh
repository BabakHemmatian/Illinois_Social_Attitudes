#!/bin/bash

#SBATCH -J Relevance_Batched_SingleProcess
#SBATCH --mail-user=babak2@illinois.edu 
#SBATCH --mail-type=ALL
#SBATCH -o relevance-batch-single-%j.out
#SBATCH -e relevance-batch-single-%j.err
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --export=ALL

export PYTHONUNBUFFERED=TRUE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $resource.py --group "$group" --years "$years"