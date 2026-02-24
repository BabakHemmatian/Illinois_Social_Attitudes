#!/bin/bash
#SBATCH -J Labeling_Batched
#SBATCH --mail-user=babak.hemmatian@stonybrook.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /home/bhemmatianbo/labeling-batch-%j.out
#SBATCH -e /home/bhemmatianbo/labeling-batch-%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --export=ALL

set -euo pipefail

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# These MUST be provided via sbatch --export=ALL,...
: "${resource:?ERROR: missing env var 'resource' (use --export=ALL,resource=...)}"
: "${group:?ERROR: missing env var 'group' (use --export=ALL,group=...)}"
: "${type:?ERROR: missing env var 'type' (use --export=ALL,type=...)}"

requires_years=("filter_keywords" "filter_language" "filter_relevance" "filter_sample" "label_moralization" "label_sentiment" "label_generalization" "label_emotion")
requires_batch=("filter_relevance" "label_moralization" "label_generalization" "label_emotion" "label_sentiment" "train_relevance")

in_array() { local needle="$1"; shift; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

ARGS=( "./code/${resource}.py" "-r" "${resource}" "-g" "${group}" "-t" "${type}" )

# If this is an array job, Slurm sets SLURM_ARRAY_TASK_ID automatically.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARGS+=( "--array" "${SLURM_ARRAY_TASK_ID}" )
fi

# Conditionally require years
if in_array "${resource}" "${requires_years[@]}"; then
  : "${years:?ERROR: env var 'years' is required for resource '${resource}'}"
  ARGS+=( "-y" "${years}" )
fi

# Conditionally require batchsize
if in_array "${resource}" "${requires_batch[@]}"; then
  : "${batchsize:?ERROR: env var 'batchsize' is required for resource '${resource}'}"
  if ! [[ "${batchsize}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: batchsize must be a positive integer" >&2
    exit 2
  fi
  ARGS+=( "-b" "${batchsize}" )
fi

# files_per_job comes from env var files_per_job
if [[ -n "${files_per_job:-}" ]]; then
  if ! [[ "${files_per_job}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: files_per_job must be a positive integer" >&2
    exit 2
  fi
  ARGS+=( "--files-per-job" "${files_per_job}" )
fi

echo "Node: $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Running: python ${ARGS[*]}"
python "${ARGS[@]}"
