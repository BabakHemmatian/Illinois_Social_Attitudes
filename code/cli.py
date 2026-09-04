### Imports

import argparse
import os
import re
import shlex
from math import ceil
from pathlib import Path
import subprocess
import sys

from utils import (
    AUTHOR_MAP_WARM_LOG,
    array_span_from_years,
    check_reqd_files,
    default_resource,
    find_latest_resource_dir,
    groups,
    init_author_file_counts_cache,
    init_author_file_counts_caches,
    init_location_cache,
    init_location_detail_cache,
    location_label_db_path,
    parse_range,
    unwarmed_files,
    validate_resource_dir,
    validate_years,
    warmed_months_from_log,
)

### Run Knobs
use_gpu = True # whether the slurm cluster version requests GPUs based on the resource type

gpu_resources = {
    "filter_relevance",
    "train_relevance",
    "label_moralization",
    "label_sentiment",
    "label_generalization",
    "label_emotion",
}

# Per-resource SLURM resource overrides (override slurm.sh defaults at submission time).
# GPU resources inherit --mem=50G from slurm.sh; CPU-only resources that need less are listed here.
RESOURCE_SLURM_RESOURCES = {
    "label_location": {"mem": "16G", "cpus-per-task": 4},
    # Streams one CSV at a time against a read-only author map; 8G/2 CPUs
    # matched the observed footprint of the full 2007-2023 runs.
    "organize_anonymize": {"mem": "8G", "cpus-per-task": 2},
    # Streams CSVs and holds one month's distinct-id set at a time.
    "verify_integrity": {"mem": "8G", "cpus-per-task": 2},
}

# The single-task author-map warm-up that cli.py chains in front of a batch
# organize_anonymize array (see _organize_anonymize_unwarmed). It holds one
# batch of months' authors in memory and runs WARM_WORKERS parallel readers
# (organize_anonymize.py); peak RSS observed at ~7 GB.
ORGANIZE_ANONYMIZE_WARM_RESOURCES = {"mem": "12G", "cpus-per-task": 4}

### Global Path Handling

dir_path = os.path.dirname(os.path.realpath(__file__))  # kept for backward-compat
CODE_DIR = Path(__file__).resolve().parent              # absolute /code
PROJECT_ROOT = CODE_DIR.parent                          # absolute project root
# ISAAC_DATA_DIR / ISAAC_MODELS_DIR redirect all reads/writes that are not
# covered by per-stage -i/-o overrides (the label_location cache dir, the
# anonymization user_map, report files). Worker scripts import these constants
# from cli, so the override propagates to every stage. Used by the workshop GUI
# to give hosted sessions isolated workspaces; defaults preserve behavior.
DATA_DIR = Path(os.environ.get("ISAAC_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = Path(os.environ.get("ISAAC_RAW_DIR", DATA_DIR / "data_reddit_raw"))
MODELS_DIR = Path(os.environ.get("ISAAC_MODELS_DIR", PROJECT_ROOT / "models"))

### Utilities

# Return a Slurm/log-file-safe slug.
def _slug(value: str) -> str:

    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")

# Build a descriptive Slurm job/log prefix from the selected CLI args.
def _build_job_tag(args) -> str:
    parts = [args.resource, args.type]
    if args.group:
        parts.append(args.group)
    if args.years:
        # Normalize the years spec before _slug runs so the tag stays readable:
        # (1) drop whitespace (a user-written '2019, 2021-2023' would otherwise
        #     leave behind '_-' artifacts after slugging), and (2) rewrite
        #     commas as underscores -- otherwise _slug turns ',' into '-' and a
        #     spec like '2019,2021-2023' becomes '2019-2021-2023', which is
        #     indistinguishable from a contiguous range. With underscores the
        #     tag reads as '2019_2021-2023', preserving the disjoint/range split.
        years_for_tag = "".join(args.years.split()).replace(",", "_")
        parts.append(years_for_tag)
    return _slug("__".join(parts))


def _shell_join(parts) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


# For a batch organize_anonymize run, resolve the input stage the same way the
# resource script will and return (unwarmed monthly input files, number of
# input files, stage name). Array tasks open the author map read-only, so a
# month whose authors have not all been assigned an ID yet must be warmed by a
# single-writer pass first; the warm log (utils.AUTHOR_MAP_WARM_LOG) records
# which months already were.
def _organize_anonymize_unwarmed(args):
    if args.input:
        input_path = validate_resource_dir(args.input, default_resource)
    else:
        input_path = find_latest_resource_dir(
            DATA_DIR / "data_reddit_curated" / args.group / args.type, default_resource
        )
    files = check_reqd_files(
        years=parse_range(args.years), type_=args.type, check_path=input_path, strict=False
    )
    warmed = warmed_months_from_log(
        PROJECT_ROOT / AUTHOR_MAP_WARM_LOG, args.group, args.type, input_path.name
    )
    return unwarmed_files(files, warmed), len(files), input_path.name


# Gets the command line arguments and returns errors if a needed argument is missing or ill-formatted
def get_args(argv=None):
    argparser = argparse.ArgumentParser(
        description="A command line interface for Illinois Social Attitudes Aggregate Corpus development and evaluation functions. See the GitHub repository's readme file for more details on the available resources."
    )

    # Conditionally require --years
    needs_years = [
        'filter_keywords',
        'filter_language',
        'filter_relevance',
        'filter_keywords_adv',
        'filter_sample',
        'label_moralization',
        'label_sentiment',
        'label_generalization',
        'label_emotion',
        'label_location',
        'organize_types',
        'organize_anonymize'
    ]

    # Conditionally require --batchsize
    needs_batchsize = [
        'filter_relevance',
        'label_moralization',
        'label_generalization',
        'label_emotion',
        'label_sentiment',
        'label_location'
    ]

    # Conditionally require --group (train_location is global, not group-specific)
    needs_group = [
        'filter_keywords',
        'filter_language',
        'filter_relevance',
        'filter_keywords_adv',
        'filter_sample',
        'metrics_interrater',
        'label_moralization',
        'label_sentiment',
        'label_generalization',
        'label_emotion',
        'label_location',
        'train_relevance',
        # Both organize_* stages resolve their input/output directories under
        # data_reddit_curated/<group>/, so a missing group fails on a path
        # rather than with a clear argparse error without this.
        'organize_types',
        'organize_anonymize',
        # Verifies whatever curated directory the group/type resolve to.
        'verify_integrity'
    ]
    argparser.add_argument(
        '-t', '--type',
        type=str,
        choices=[
            'submissions',
            'comments',
            'all',
        ],
        required=False,
        help="Indicate the type of Reddit post (submission, comment, or all) you want processed. 'all' is implemented for 'filter_sample', 'train' and 'organize' resources. For other resources, you can use 'organize_types' to aggregate outputs post-hoc. Not required for 'organize_types', which always merges comments and submissions into an 'all' output."
    )
    argparser.add_argument(
        '-c', '--sample',
        type=int,
        dest='sample',
        help='Per-annotator target document count for filter_sample. The realized total matches this exactly when data isn\'t sparse; shortfalls in a year only narrow the gap by that year\'s contribution.'
    )
    argparser.add_argument(
        '-n', '--num-annotators',
        type=int,
        dest='num_annotators',
        default=2,
        help='Number of annotators that filter_sample should produce sample files for. Default: 2. Also used by metrics_interrater to know how many rater files to load.'
    )
    argparser.add_argument(
        '-p', '--perc-overlap',
        type=float,
        dest='perc_overlap',
        default=1.0,
        help='Fraction of each annotator\'s samples that should be shared (same docs, same random_id) with every other annotator. 1.0 (default) = every annotator gets the same set in a different shuffle; 0.0 = annotators get fully disjoint sets; 0.1 = 10%% of each annotator\'s set is shared, 90%% is annotator-specific. Only applies to filter_sample.'
    )
    argparser.add_argument(
        '-S', '--sample-target',
        type=str,
        dest='target',
        choices=[
            'filter_keywords', 'filter_language','filter_relevance', 'filter_keywords_adv', 'label_moralization',
            'label_generalization', 'label_sentiment', 'label_emotion', 'label_location'
        ],
        help='Identifies the resource from whose outputs filter_sample is to extract a subset of documents. Only applicable to filter and label resources.'
    )
    argparser.add_argument(
        '--stratify',
        type=str,
        dest='stratify',
        choices=['auto', 'on', 'off'],
        default='auto',
        help="filter_sample only: control top/bottom/random keyword-count stratification. "
             "'auto' (default) stratifies only for filter_* targets and is fully random for "
             "label_*/organize_* targets. 'on' forces stratification regardless of target — "
             "use this to draw filter-style samples from label_ outputs that still carry the "
             "keyword column (index 7). 'off' forces a fully random sample."
    )
    argparser.add_argument(
        '-i', '--input',
        type=str,
        help="The input folder for the resource. Defaults to the order of resources indicated in the repository."
    )
    argparser.add_argument(
        '-o','--output',
        type=str,
        help="Optionally identify an output folder for the resource. If not provided, defaults to the order of resources indicated in the repository."
    )
    argparser.add_argument(
        '-r', '--resource',
        type=str,
        choices=[
            'filter_keywords', 'filter_language', 'filter_sample',
            'filter_relevance', 'filter_keywords_adv', 'metrics_interrater', 'label_moralization',
            'label_generalization', 'label_sentiment', 'label_emotion', 'label_location','organize_types','organize_anonymize',
            'train_relevance', 'train_location_preprocess', 'train_location_training','train_location_weighting',
            'verify_integrity'
        ],
        required=True,
        help="Indicate the type of processing needed (see repository). 'filter_keywords' should be run first. 'organize' resources depend on 'filter'/'label' processed data files. 'verify_integrity' can be run after any stage: it reads every byte of the most advanced curated directory for the group/type (or of --input) and, for organize_ outputs, reconciles row counts against their inputs and anonymized author IDs against the author map. Counterparts are located by the canonical data_reddit_curated/<group>/<type>/<stage> layout relative to the verified directory (an _anon directory's source is its sibling <stage>; an 'all' directory's inputs are ../../comments/<stage> and ../../submissions/<stage>); a missing counterpart skips only that reconciliation and is noted in the report. Any file needing attention is listed as 'Rebuild <file>' and the run exits non-zero."
    )
    argparser.add_argument(
        '-g', '--group',
        type=str,
        choices=list(groups.keys()),
        required=False,
        help='Identify the social group to which the processing should be applied. Not required for train_location.'
    )
    argparser.add_argument(
        '-y', '--years',
        type=str,
        help='Determine the years to which the tool should be applied for the indicated groups. Accepts a single year (e.g. 2019), a contiguous range with a dash (e.g. 2019-2023), or any comma-separated combination of those (e.g. 2007,2009,2011-2017). All years must fall between 2007 and 2023.'
    )
    argparser.add_argument(
        '-b', '--batchsize',
        type=int,
        help="Enter an integer for the neural network batch size. Required for filter_relevance and all the labeling resources.",
    )
    argparser.add_argument(
        '-s', '--slurm',
        action="store_true",
        help="Submit a Slurm job. Best used for NN resources (filter_relevance, label_moralization, label_generalization). Should only be used on a Slurm computing cluster. Resources that need a preparatory or follow-up single-task step queue it automatically as a dependency chain: organize_anonymize warms the author map before its array (skipped when the requested months are already warmed) and label_location merges scan-state shards after its array."
    )
    argparser.add_argument(
        '-j', "--num-jobs",
        dest='numjob',
        type=int,
        default=10,
        help="The cap on the number of simultaneous jobs spawned if the slurm flag is raised."
    )
    argparser.add_argument(
        "--mem",
        dest='mem',
        type=str,
        help="Override per-task memory for the Slurm submission (e.g. '8G', '16000M'). Falls back to the per-resource default in RESOURCE_SLURM_RESOURCES, then to slurm.sh."
    )
    argparser.add_argument(
        "--cpus-per-task",
        dest='cpus_per_task',
        type=int,
        help="Override per-task CPU count for the Slurm submission. Falls back to the per-resource default in RESOURCE_SLURM_RESOURCES, then to slurm.sh."
    )
    argparser.add_argument(
        "--files-per-job",
        type=int,
        default=1,
        help="Number of monthly files each Slurm array task should process."
    )
    argparser.add_argument(
        "--quick",
        dest="quick",
        action="store_true",
        help="verify_integrity only: inspect each file's head and tail instead of reading every byte. Catches truncation and header damage quickly; skips row-count reconciliation and the author-map cross-reference."
    )
    argparser.add_argument(
        "--array",
        type=int,
        help="Index from SLURM_ARRAY_TASK_ID; if set, process only that indexed file. If omitted, process all files."
    )
    argparser.add_argument(
        "--dependency",
        dest="dependency",
        default=None,
        help="Forwarded verbatim to sbatch --dependency (e.g. 'afterany:43472') "
             "so this submission waits for another job to reach a terminal state "
             "instead of running concurrently and adding contention."
    )
    argparser.add_argument(
        "--array-order",
        dest="array_order",
        default=None,
        help="Path to a file (or inline comma/space list) of file_list indices. "
             "When set, the SLURM array slot indexes THIS list instead of "
             "file_list directly, so concurrent tasks under %%cap can be spread "
             "far apart (less cache-DB lock contention and no duplicate raw-file "
             "decompression). The array span is auto-set to 0..len-1. Pass a "
             "file path for SLURM runs (the value is forwarded via --export, "
             "which cannot carry commas)."
    )
    argparser.add_argument(
        "--maxitems", "--max-items", "--max_items_per_author",
        dest="maxitems",
        type=int,
        help="Max number of comments/submissions sampled per author for location estimation (default 25)."
    )
    argparser.add_argument(
        "--maxfiles", "--max-files", "--max_files_to_scan",
        dest="maxfiles",
        type=int,
        help="Hard cap on the number of monthly files scanned while collecting samples (default 60)."
    )
    argparser.add_argument(
        "--maxradius", "--max-radius", "--max_radius",
        dest="maxradius",
        type=int,
        help="Max month-radius around target month to consider while scanning (default 30)."
    )
    argparser.add_argument(
        "--input_2", "-2",
        dest="input_2",
        type=str,
        help="The second input folder for 'organize_types' and 'train_location_weighting'. For organize_types, one input should be a 'comments' and the other a 'submissions' folder. For train_location_weighting, 'input' should be the preprocessed features folder and 'input_2' the regression model folder."
    )

    args = argparser.parse_args(argv)

    # --type is meaningful for every resource except organize_types, which always
    # merges comments and submissions into a single 'all' output. Default it there
    # so downstream code (job tags, slurm/-t forwarding) keeps a valid value; for
    # every other resource --type remains required.
    if args.resource == "organize_types":
        if args.type is None:
            args.type = "all"
    elif args.type is None:
        argparser.error("--type is required for this resource")

    # Restrict -t all to the location training resources only.
    if args.type == "all" and not any(k in args.resource for k in ("train", "organize", "sample", "verify")):
        argparser.error("--type all is only valid for filter_sample as well as train/organize/verify resources")

    # Validate group if required
    if args.resource in needs_group and not args.group:
        argparser.error("--group is required for this resource")

    # Validate years if required
    if args.resource in needs_years:
        if not args.years:
            argparser.error("--years is required for this resource")
        validate_years(args.years, argparser)

    # Validate batchsize if required
    if args.resource in needs_batchsize:
        if args.batchsize is None:
            argparser.error("--batchsize is required for this resource")
        if args.batchsize <= 0:
            argparser.error("--batchsize must be a positive integer")

    if args.files_per_job <= 0:
        argparser.error("--files-per-job must be a positive integer")

    if args.num_annotators is not None and args.num_annotators < 1:
        argparser.error("--num-annotators must be at least 1")
    if args.perc_overlap is not None and not (0.0 <= args.perc_overlap <= 1.0):
        argparser.error("--perc-overlap must be between 0.0 and 1.0 inclusive")

    return args


# evaluate the entered arguments based on requirements and whether the 'slurm' flag is raised
if __name__ == "__main__":
    args = get_args()

    # Pre-initialize the SQLite caches once from this single process so that
    # parallel Slurm array tasks (or local ProcessPoolExecutor workers) do not
    # race on the first WAL-mode setup, which can raise "database is locked".
    # The cache is per-type and group-global (shared across all six social
    # groups within a type) so that authors cross-pollinated by different
    # groups don't trigger redundant raw scans.
    if args.resource == "label_location":
        location_cache_dir = DATA_DIR / "data_reddit_curated" / "data_reddit_location"
        location_cache_dir.mkdir(parents=True, exist_ok=True)
        # Label tables (author_location + author_location_detail) live in one DB,
        # keyed by author so cross-year/-group dedup is preserved. The large,
        # regenerable author_file_counts table is sharded into one DB per year.
        label_db_path = location_label_db_path(str(location_cache_dir), args.type)
        init_location_cache(label_db_path)
        init_location_detail_cache(label_db_path)
        # Pre-create the per-year file_counts DBs for the requested years from
        # this single process so parallel array tasks don't race on CREATE TABLE.
        years_list = parse_range(args.years) if args.years else []
        init_author_file_counts_caches(str(location_cache_dir), args.type, years_list)

    if args.slurm:
        slurm_vars = [f"resource={args.resource}", f"type={args.type}"]
        array_spec = None

        array_resources = {
            "filter_keywords",
            "filter_language",
            "filter_relevance",
            "filter_keywords_adv",
            "label_moralization",
            "label_sentiment",
            "label_generalization",
            "label_emotion",
            "label_location",
            # Both organize_* stages select their months by the YYYY-MM parsed
            # from each filename rather than by position in the file list, so a
            # Slurm array slot maps to a fixed month regardless of gaps.
            "organize_types",
            "organize_anonymize",
            # verify_integrity also selects by YYYY-MM; without --years it runs
            # as one job over every month present (no array_spec is built).
            "verify_integrity",
        }

        if args.group:
            slurm_vars.append(f"group={args.group}")
        if args.years:
            slurm_vars.append(f"years={args.years}")

            if args.resource in array_resources:
                months = array_span_from_years(args.years)
                num_jobs = ceil(months / args.files_per_job)
                array_spec = f"0-{num_jobs - 1}"
                # --array-order overrides the span: one slot per listed index.
                if getattr(args, "array_order", None):
                    order_path = Path(args.array_order)
                    if order_path.exists():
                        raw = order_path.read_text()
                    else:
                        raw = args.array_order
                    n_order = len([t for t in re.split(r"[,\s]+", raw.strip()) if t])
                    if n_order:
                        array_spec = f"0-{n_order - 1}"
                    slurm_vars.append(f"array_order={args.array_order}")

        if args.batchsize:
            slurm_vars.append(f"batchsize={args.batchsize}")
        if args.files_per_job:
            slurm_vars.append(f"files_per_job={args.files_per_job}")

        if args.sample is not None:
            slurm_vars.append(f"sample={args.sample}")
        if args.target is not None:
            slurm_vars.append(f"target={args.target}")
        if args.resource in ("filter_sample", "metrics_interrater") and args.num_annotators is not None:
            slurm_vars.append(f"num_annotators={args.num_annotators}")
        if args.resource == "filter_sample" and args.perc_overlap is not None:
            slurm_vars.append(f"perc_overlap={args.perc_overlap}")
        if args.resource == "filter_sample" and getattr(args, "stratify", "auto") != "auto":
            slurm_vars.append(f"stratify={args.stratify}")
        if args.resource == "verify_integrity" and getattr(args, "quick", False):
            slurm_vars.append("quick=1")

        # Location-labeling sampling controls (forwarded to label_location)
        if getattr(args, "maxitems", None) is not None:
            slurm_vars.append(f"maxitems={args.maxitems}")
        if getattr(args, "maxfiles", None) is not None:
            slurm_vars.append(f"maxfiles={args.maxfiles}")
        if getattr(args, "maxradius", None) is not None:
            slurm_vars.append(f"maxradius={args.maxradius}")

        # Forward optional path overrides to slurm.sh
        if args.input:
            slurm_vars.append(f"input={args.input}")
        if args.input_2:
            slurm_vars.append(f"input_2={args.input_2}")
        if args.output:
            slurm_vars.append(f"output={args.output}")

        slurm_script = CODE_DIR / "slurm.sh"
        concurrency_cap = args.numjob  # number of simultaneous tasks
        array_flag = f"{array_spec}%{concurrency_cap}" if array_spec else None

        log_dir = PROJECT_ROOT / "slurm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        job_tag = _build_job_tag(args)
        log_token = "%A_%a" if array_spec else "%j"
        stdout_path = log_dir / f"{job_tag}__{log_token}.out"
        stderr_path = log_dir / f"{job_tag}__{log_token}.err"

        # Optional Slurm dependency so a new chain waits for a running job to
        # finish instead of adding concurrent load (e.g. afterany:<jobid> to
        # start only once a prior array reaches a terminal state). A resource
        # with an automatic preparatory step (below) hands this to that step and
        # chains its own array to the step instead.
        dependency = getattr(args, "dependency", None)

        # organize_anonymize: array tasks open the author map read-only, so
        # every author in the requested months must already hold an ID. If the
        # warm log shows unwarmed months, queue a SINGLE warm-up job first --
        # ORGANIZE_ANONYMIZE_WARM=1 puts organize_anonymize.py into warm mode,
        # which assigns the missing IDs in bulk and writes no anonymized output
        # -- and chain the array to it with afterok, so a failed warm-up holds
        # the array back instead of letting every task fail on the read-only
        # map. Local (non-Slurm) runs mint IDs on demand and never need this.
        if args.resource == "organize_anonymize" and array_flag:
            pending, n_inputs, stage = _organize_anonymize_unwarmed(args)
            if pending:
                print(f"[cli] author map: {len(pending)} of {n_inputs} requested {args.type} month(s) "
                      f"for {args.group}/{stage} not yet warmed; queuing a single-task warm-up job "
                      f"and chaining the anonymization array to it (afterok)")
                warm_keys = ("resource", "type", "group", "years", "input", "output")
                warm_vars = [v for v in slurm_vars if v.split("=", 1)[0] in warm_keys]
                warm_vars.append("ORGANIZE_ANONYMIZE_WARM=1")
                warm_tag = f"{job_tag}__warm"
                warm_parts = [
                    "sbatch", "--parsable",
                    "--job-name", warm_tag,
                    "--output", str(log_dir / f"{warm_tag}__%j.out"),
                    "--error", str(log_dir / f"{warm_tag}__%j.err"),
                    "--export", f"ALL,{','.join(warm_vars)}",
                    "--mem", ORGANIZE_ANONYMIZE_WARM_RESOURCES["mem"],
                    "--cpus-per-task", str(ORGANIZE_ANONYMIZE_WARM_RESOURCES["cpus-per-task"]),
                ]
                if dependency:
                    warm_parts.extend(["--dependency", str(dependency)])
                warm_parts.append(str(slurm_script))
                print(f"[cli] submitting: {_shell_join(warm_parts)}")
                wres = subprocess.run(warm_parts, capture_output=True, text=True)
                sys.stdout.write(wres.stdout)
                if wres.returncode != 0:
                    sys.stderr.write(wres.stderr)
                    raise SystemExit(wres.returncode)
                warm_job_id = wres.stdout.strip().splitlines()[-1].split(";")[0]
                dependency = f"afterok:{warm_job_id}"
            else:
                print(f"[cli] author map already warmed for all {n_inputs} requested {args.type} month(s) "
                      f"for {args.group}/{stage}; submitting the anonymization array directly")

        cmd_parts = [
            "sbatch",
            "--job-name", job_tag,
            "--output", str(stdout_path),
            "--error", str(stderr_path),
            "--export", f"ALL,{','.join(slurm_vars)}",
        ]

        if dependency:
            cmd_parts.extend(["--dependency", str(dependency)])

        if args.resource in gpu_resources and use_gpu:
            cmd_parts.extend(["--gres", "gpu:1"])

        slurm_res = RESOURCE_SLURM_RESOURCES.get(args.resource, {})
        mem = args.mem if args.mem is not None else slurm_res.get("mem")
        cpus_per_task = args.cpus_per_task if args.cpus_per_task is not None else slurm_res.get("cpus-per-task")
        if mem is not None:
            cmd_parts.extend(["--mem", str(mem)])
        if cpus_per_task is not None:
            cmd_parts.extend(["--cpus-per-task", str(cpus_per_task)])

        if array_flag:
            cmd_parts.extend(["--array", array_flag])
        cmd_parts.append(str(slurm_script))

        # Submit the array, capturing its job id (--parsable) so label_location
        # can chain an automatic post-array merge below.
        submit_parts = cmd_parts[:1] + ["--parsable"] + cmd_parts[1:]
        print(f"[cli] submitting: {_shell_join(submit_parts)}")
        res = subprocess.run(submit_parts, capture_output=True, text=True)
        sys.stdout.write(res.stdout)
        if res.returncode != 0:
            sys.stderr.write(res.stderr)
            raise SystemExit(res.returncode)
        array_job_id = res.stdout.strip().splitlines()[-1].split(";")[0] if res.stdout.strip() else ""

        # label_location: automatically queue a SINGLE post-array merge that
        # folds the per-task scan-state shards (written automatically by every
        # array task) into the canonical year DBs. --dependency=afterany so it
        # ALSO runs after wall-time-killed tasks -- those killed-task shards are
        # exactly the banked progress we must rescue, and an in-task merge would
        # miss them. It is a single writer with nothing else running, so no NFS
        # write contention; LABEL_LOCATION_MERGE=1 puts label_location.py into
        # merge mode (GPU-free, no --array). The merge job's own guard waits out
        # any other still-running same-type array.
        if args.resource == "label_location" and array_flag and array_job_id:
            # slurm.sh requires resource/type/years/batchsize for label_location
            # (and label_location.py parses them at import); merge mode then
            # ignores everything but the cache paths. Forward exactly those.
            merge_keys = ("resource", "type", "group", "years", "batchsize")
            merge_vars = [v for v in slurm_vars if v.split("=", 1)[0] in merge_keys]
            merge_vars.append("LABEL_LOCATION_MERGE=1")
            merge_tag = f"{job_tag}__merge"
            merge_parts = [
                "sbatch", "--parsable",
                "--job-name", merge_tag,
                "--output", str(log_dir / f"{merge_tag}__%j.out"),
                "--error", str(log_dir / f"{merge_tag}__%j.err"),
                "--export", f"ALL,{','.join(merge_vars)}",
                "--dependency", f"afterany:{array_job_id}",
                "--mem", "8G", "--cpus-per-task", "1",
                str(slurm_script),
            ]
            print(f"[cli] queuing post-array merge: {_shell_join(merge_parts)}")
            mres = subprocess.run(merge_parts, capture_output=True, text=True)
            sys.stdout.write(mres.stdout)
            if mres.returncode != 0:
                sys.stderr.write("[cli] WARNING: post-array merge submit failed; run it manually after the array:\n")
                sys.stderr.write("[cli]   LABEL_LOCATION_MERGE=1 python code/label_location.py "
                                 f"-r label_location -t {args.type} -g {args.group} -y {args.years} -b {args.batchsize}\n")
                sys.stderr.write(mres.stderr)
    else:
        # Robust path to the resource script inside code/
        resource_script = CODE_DIR / f"{args.resource}.py"
        cmd_parts = [
            sys.executable,
            str(resource_script),
            "-t", args.type,
            "-r", args.resource,
        ]
        if args.group:
            cmd_parts.extend(["-g", args.group])
        if args.years:
            cmd_parts.extend(["-y", args.years])
        if args.batchsize:
            cmd_parts.extend(["-b", str(args.batchsize)])
        # Forward array index and location-labeling knobs when running locally
        if args.array is not None:
            cmd_parts.extend(["--array", str(args.array)])
        if getattr(args, "array_order", None):
            cmd_parts.extend(["--array-order", str(args.array_order)])
        if args.sample is not None:
            cmd_parts.extend(["-c", str(args.sample)])
        if args.target is not None:
            cmd_parts.extend(["-S", args.target])
        if args.resource in ("filter_sample", "metrics_interrater") and args.num_annotators is not None:
            cmd_parts.extend(["-n", str(args.num_annotators)])
        if args.resource == "filter_sample" and args.perc_overlap is not None:
            cmd_parts.extend(["-p", str(args.perc_overlap)])
        if args.resource == "filter_sample" and getattr(args, "stratify", "auto") != "auto":
            cmd_parts.extend(["--stratify", args.stratify])
        if args.resource == "verify_integrity" and getattr(args, "quick", False):
            cmd_parts.append("--quick")
        if getattr(args, "maxitems", None) is not None:
            cmd_parts.extend(["--maxitems", str(args.maxitems)])
        if getattr(args, "maxfiles", None) is not None:
            cmd_parts.extend(["--maxfiles", str(args.maxfiles)])
        if getattr(args, "maxradius", None) is not None:
            cmd_parts.extend(["--maxradius", str(args.maxradius)])
        if args.files_per_job:
            cmd_parts.extend(["--files-per-job", str(args.files_per_job)])

        # Forward optional path overrides when running locally
        if args.input:
            cmd_parts.extend(["-i", args.input])
        if args.input_2:
            cmd_parts.extend(["-2", args.input_2])
        if args.output:
            cmd_parts.extend(["-o", args.output])

        # Pretty log line only
        print("[cli] running:", subprocess.list2cmdline([str(p) for p in cmd_parts]))

        # Cross-platform execution without shell-quoting issues
        subprocess.run(cmd_parts, check=True)