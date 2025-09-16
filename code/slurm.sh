#!/bin/bash
#SBATCH -job-name=Labeling_Batched
#SBATCH --mail-user=bhemmatian2@unl.edu
#SBATCH --mail-type=ALL
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -o labeling-batch-%j.out
#SBATCH -e labeling-batch-%j.err
#SBATCH --mem=4gb 
#SBATCH --ntasks-per-node=2 
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --array=0-1

module load cuda

set -euo pipefail

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# resource, group, years, batchsize are expected from: sbatch --export=ALL,resource=...,group=...,years=...,batchsize=... slurm.sh

# Which resources require each optional arg (keep in sync with the Python CLI)
requires_years=("filter_keywords" "filter_language" "filter_relevance" "filter_sample" "label_moralization" "label_sentiment" "label_generalization" "label_emotion")
requires_batch=("filter_relevance" "label_moralization" "label_generalization" "label_emotion")

in_array() {
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

# Base args
ARGS=( "./code/${resource}.py" "-r" "${resource}" "-g" "${group}" "--array" "${SLURM_ARRAY_TASK_ID}" )

# Conditionally add --years (and enforce if required)
if in_array "${resource}" "${requires_years[@]}"; then
  if [[ -z "${years:-}" ]]; then
    echo "ERROR: --years is required for resource '${resource}'" >&2
    exit 2
  fi
  ARGS+=( "-y" "${years}" )
elif [[ -n "${years:-}" ]]; then
  # years provided but not required — you can choose to allow or ignore.
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
elif [[ -n "${batchsize:-}" ]]; then
  # batchsize provided but not required — allow it, or comment out to ignore
  if ! [[ "${batchsize}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --batchsize must be a positive integer" >&2
    exit 2
  fi
  ARGS+=( "-b" "${batchsize}" )
fi

# Run
python "${ARGS[@]}"
