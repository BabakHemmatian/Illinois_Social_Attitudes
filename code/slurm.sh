#!/bin/bash
#SBATCH --mail-user=[type your email here]
#SBATCH --mail-type=END,FAIL
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# --cpus-per-task=8
#SBATCH --export=ALL

set -euo pipefail

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

requires_years=("filter_keywords" "filter_language" "filter_relevance" "filter_keywords_adv" "filter_sample" "label_moralization" "label_sentiment" "label_generalization" "label_emotion" "label_location")
requires_batch=("filter_relevance" "label_moralization" "label_generalization" "label_emotion" "label_sentiment" "label_location")

in_array() { local needle="$1"; shift; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

build_task_label() {
  if [[ -z "${years:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    return 0
  fi

  python - "${years}" "${files_per_job:-1}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import sys

years = sys.argv[1]
files_per_job = max(int(sys.argv[2]), 1)
task_id = int(sys.argv[3])

if "-" in years:
    start_year, end_year = map(int, years.split("-", 1))
else:
    start_year = end_year = int(years)

months = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        months.append(f"{year:04d}-{month:02d}")

start_idx = task_id * files_per_job
end_idx = min(start_idx + files_per_job, len(months))
chunk = months[start_idx:end_idx]

if not chunk:
    print(f"task{task_id}")
elif len(chunk) == 1:
    print(chunk[0])
else:
    print(f"{chunk[0]}_to_{chunk[-1]}")
PY
}

# Base args
ARGS=( "./code/${resource}.py" "-r" "${resource}" "-t" "${type}" )

if [[ -n "${group:-}" ]]; then
  ARGS+=( "-g" "${group}" )
fi
if [[ -n "${sample:-}" ]]; then
  ARGS+=( "-c" "${sample}" )
fi

if [[ -n "${target:-}" ]]; then
  ARGS+=( "-S" "${target}" )
fi

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

# Pass files-per-job if set
if [[ -n "${files_per_job:-}" ]]; then
  ARGS+=( "--files-per-job" "${files_per_job}" )
fi

# Forward location-labeling sampling controls when present
if [[ -n "${maxitems:-}" ]]; then
  ARGS+=( "--maxitems" "${maxitems}" )
fi
if [[ -n "${maxfiles:-}" ]]; then
  ARGS+=( "--maxfiles" "${maxfiles}" )
fi
if [[ -n "${maxradius:-}" ]]; then
  ARGS+=( "--maxradius" "${maxradius}" )
fi

# Update the visible Slurm job name for array tasks so squeue reflects the concrete month span.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  name_parts=("${resource}" "${type}")
  if [[ -n "${group:-}" ]]; then
    name_parts+=("${group}")
  fi
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_label="$(build_task_label)"
    if [[ -n "${task_label}" ]]; then
      name_parts+=("${task_label}")
    fi
  elif [[ -n "${years:-}" ]]; then
    name_parts+=("${years}")
  fi

  job_name="$(IFS=__ ; echo "${name_parts[*]}")"
  scontrol update JobId="${SLURM_JOB_ID}" JobName="${job_name}" >/dev/null 2>&1 || true
fi

echo "Running: python ${ARGS[*]}"
python "${ARGS[@]}"
