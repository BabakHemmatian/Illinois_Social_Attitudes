#!/bin/bash
#SBATCH --mail-user=babak.hemmatian@stonybrook.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=96:00:00
#SBATCH --mem=32G
# --cpus-per-task=8
#SBATCH --export=ALL

set -euo pipefail

# Activate the project's conda env (the README documents creating one named
# ISAAC). On HPC clusters with Environment Modules / Lmod, 'module load conda'
# is tried first; on clusters where conda is already on PATH (system install
# or user-init in ~/.bashrc), the module step is a silent no-op. The activate
# step is what actually puts the project's python on PATH, so an sbatch
# launched from any shell (interactive or not) works without the caller
# having to 'conda activate ISAAC' first.
if command -v module >/dev/null 2>&1; then
    module load conda >/dev/null 2>&1 || true
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "[slurm.sh] ERROR: 'conda' not found on PATH. See README.md for ISAAC env setup." >&2
    exit 1
fi
# Conda's activation hooks reference some unset shell variables (e.g.
# ADDR2LINE in binutils' activation hook), which trips 'set -u'. Disable
# nounset around the activation, then restore it.
set +u
eval "$(conda shell.bash hook)"
# Pop any conda envs inherited from the submitter shell (--export=ALL can
# propagate CONDA_DEFAULT_ENV/CONDA_PREFIX). Without this, 'conda activate
# ISAAC' short-circuits as a no-op when ISAAC is already marked active, and
# ISAAC/bin can end up behind miniforge3/bin in PATH -> wrong python.
while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate
done
conda activate ISAAC
set -u

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configure GPUs if allocated by Slurm
if [[ -n "${SLURM_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_GPUS_ON_NODE:-0}"
  echo "[slurm.sh] GPU allocation detected: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

requires_years=("filter_keywords" "filter_language" "filter_relevance" "filter_keywords_adv" "filter_sample" "label_moralization" "label_sentiment" "label_generalization" "label_emotion" "label_location" "organize_types" "organize_anonymize")
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

# Mirror utils.parse_range: accept comma-separated combinations of single
# years and contiguous ranges (e.g. "2019", "2019-2023", "2007,2009,2011-2017"),
# returning a sorted/deduplicated list. Keeping this in lock-step with
# parse_range is what guarantees the array-index -> month mapping stays aligned
# with the file_list each python script computes internally.
parsed_years = set()
for tok in years.split(","):
    tok = tok.strip()
    if not tok:
        continue
    if "-" in tok:
        s, e = map(int, tok.split("-", 1))
    else:
        s = e = int(tok)
    parsed_years.update(range(s, e + 1))
parsed_years = sorted(parsed_years)

months = [f"{y:04d}-{m:02d}" for y in parsed_years for m in range(1, 13)]

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

if [[ -n "${num_annotators:-}" ]]; then
  ARGS+=( "-n" "${num_annotators}" )
fi

if [[ -n "${perc_overlap:-}" ]]; then
  ARGS+=( "-p" "${perc_overlap}" )
fi

if [[ -n "${stratify:-}" ]]; then
  ARGS+=( "--stratify" "${stratify}" )
fi

# Forward optional input/output overrides
if [[ -n "${input:-}" ]]; then
  ARGS+=( "-i" "${input}" )
fi
if [[ -n "${input_2:-}" ]]; then
  ARGS+=( "-2" "${input_2}" )
fi
if [[ -n "${output:-}" ]]; then
  ARGS+=( "-o" "${output}" )
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
  # NOTE: The scontrol command was hanging on the particular cluster we used, hence the commenting out.
  # scontrol update JobId="${SLURM_JOB_ID}" JobName="${job_name}" >/dev/null 2>&1 || true
fi

echo "Running: python ${ARGS[*]}"

# Run python, mirroring stderr into a temp file so we can post-mortem on
# transient node-side CUDA/NVML driver failures. If detected (and we haven't
# already requeued too many times), call `scontrol requeue` so SLURM puts
# the job back in the queue, almost certainly landing it on a different
# node. afterok dependents stay PD while the requeued attempt runs, instead
# of getting cascade-cancelled by --kill-on-invalid-dep=yes.
#
# Only the known transient patterns trigger a requeue; real code failures
# fall through with the original exit code so they surface as FAILED.
ERR_TMP="$(mktemp -t isaac_slurm_stderr.XXXXXX)"
# shellcheck disable=SC2064
trap "rm -f '${ERR_TMP}'" EXIT

set +e
python "${ARGS[@]}" 2> >(tee -a "${ERR_TMP}" >&2)
PY_RC=$?
set -e

TRANSIENT_RE="Can't initialize NVML|NVML_SUCCESS == DriverAPI|INTERNAL ASSERT FAILED.*CUDACachingAllocator|CUDA error: no CUDA-capable device"
MAX_REQUEUES=2
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"

if [[ "${PY_RC}" -ne 0 ]] \
   && [[ "${RESTART_COUNT}" -lt "${MAX_REQUEUES}" ]] \
   && [[ -n "${SLURM_JOB_ID:-}" ]] \
   && grep -qE "${TRANSIENT_RE}" "${ERR_TMP}" 2>/dev/null; then
  NODE="${SLURMD_NODENAME:-unknown}"
  echo "[slurm.sh] Detected transient CUDA/NVML failure on node ${NODE} (restart=${RESTART_COUNT}/${MAX_REQUEUES}); requeueing ${SLURM_JOB_ID}" >&2
  if scontrol requeue "${SLURM_JOB_ID}"; then
    # scontrol requeue sends SIGTERM shortly after returning; the lines below
    # may not execute. Exit 0 just in case it doesn't, so we don't trip
    # --kill-on-invalid-dep=yes between the requeue call and the SIGTERM.
    sleep 5
    exit 0
  else
    echo "[slurm.sh] scontrol requeue failed; surfacing original exit code ${PY_RC}" >&2
  fi
fi

exit "${PY_RC}"
