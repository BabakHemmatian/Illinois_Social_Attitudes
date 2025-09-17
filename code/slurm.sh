#!/bin/bash
#SBATCH -J Labeling_Batched
#SBATCH --mail-user=bhemmatian2@unl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o labeling-batch-%j.out
#SBATCH -e labeling-batch-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=64000
#SBATCH --export=ALL

set -euo pipefail

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# resource, group, years, batchsize are expected from:
# sbatch --array=... --export=ALL,resource=...,group=...,years=...,batchsize=... slurm.sh

requires_years=("filter_keywords" "filter_language" "filter_relevance" "filter_sample" "label_moralization" "label_sentiment" "label_generalization" "label_emotion")
requires_batch=("filter_relevance" "label_moralization" "label_generalization" "label_emotion")

in_array() { local needle="$1"; shift; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

# Base args
ARGS=( "./code/${resource}.py" "-r" "${resource}" "-g" "${group}" )

# Only pass --array if Slurm provided it
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARGS+=( "--array" "${SLURM_ARRAY_TASK_ID}" )
fi

# Conditionally add --years (and enforce if required)
if in_array "${resource}" "${requires_years[@]}"; then
  if [[ -z "${years:-}" ]]; then
    echo "ERROR: --years is required for resource '${resource}'" >&2
    exit 2
  fi
  ARGS+=( "-y" "${years}" )
fi

# Conditionally add --batchsize (and enforce positive integer)
if in_array "${resource}" "${requires_batch[@]}"; then
  if [[ -z "${batchsize:-}" ]]; then
    echo "ERROR: --batchsize is required for resource '${resource}'" >&2
    exit 2
  fi
  if ! [[ "${batchsize}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --batchsize must be a positive integer" >&2
    exit 2
  fi
  ARGS+=( "-b" "${batchsize}" )
fi

echo "Running: python ${ARGS[*]}"
python "${ARGS[@]}"
