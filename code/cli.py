import argparse
import os
import re
import shlex
from math import ceil
from pathlib import Path
import subprocess
import sys

from utils import groups, validate_years, array_span_from_years

# set path variables
dir_path = os.path.dirname(os.path.realpath(__file__))  # kept for backward-compat
CODE_DIR = Path(__file__).resolve().parent              # absolute /code
PROJECT_ROOT = CODE_DIR.parent                          # absolute project root

# Return a Slurm/log-file-safe slug.
def _slug(value: str) -> str:
    
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")

# Build a descriptive Slurm job/log prefix from the selected CLI args.
def _build_job_tag(args) -> str:
    parts = [args.resource, args.type]
    if args.group:
        parts.append(args.group)
    if args.years:
        parts.append(args.years)
    return _slug("__".join(parts))


def _shell_join(parts) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


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
        'label_location'
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
        'train_relevance'
    ]
    argparser.add_argument(
        '-t', '--type',
        type=str,
        choices=[
            'submissions',
            'comments'
        ],
        required=True,
        help="Indicate the type of Reddit post (submission or comment) you want processed."
    )
    argparser.add_argument(
        '-c', '--sample',
        type=int,
        dest='sample',
        help='the number of documents per annotator that filter_sample should aim for. Note that because of stratification, the final count might be substantially different.'
    )
    argparser.add_argument(
        '-S', '--sample-target',
        type=str,
        dest='target',
        choices=[
            'filter_keywords', 'filter_language','filter_relevance', 'filter_keywords_adv', 'label_moralization',
            'label_generalization', 'label_sentiment', 'label_emotion', 'label_location'
        ],
        help='Identifies the resource from whose outputs filter_sample is to extract a subset of documents. Only valid for filter and label resources.'
    )
    argparser.add_argument(
        '-r', '--resource',
        type=str,
        choices=[
            'filter_keywords', 'filter_language', 'filter_sample',
            'filter_relevance', 'filter_keywords_adv', 'metrics_interrater', 'label_moralization',
            'label_generalization', 'label_sentiment', 'label_emotion', 'label_location', 'train_relevance', 'train_location'
        ],
        required=True,
        help="Indicate the type of processing needed. Labeling and metrics options require the output of filtering steps for the indicated years. Filtering should be done with consecutive commands in order: keywords, language, then relevance."
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
        help='Determine the range of years to which the tool should be applied for the indicated groups. Must be either a number between 2007 and 2023 or a range in that time frame with the start and end separated by a dash.'
    )
    argparser.add_argument(
        '-b', '--batchsize',
        type=int,
        help="Enter an integer for the neural network batch size. Required for filter_relevance and all the labeling resources.",
    )
    argparser.add_argument(
        '-s', '--slurm',
        action="store_true",
        help="Submit a Slurm job. Best used for NN resources (filter_relevance, label_moralization, label_generalization). Should only be used on a Slurm computing cluster."
    )
    argparser.add_argument(
        '-j', "--num-jobs",
        dest='numjob',
        type=int,
        default=10,
        help="The cap on the number of simultaneous jobs spawned if the slurm flag is raised."
    )
    argparser.add_argument(
        "--files-per-job",
        type=int,
        default=1,
        help="Number of monthly files each Slurm array task should process."
    )
    argparser.add_argument(
        "--array",
        type=int,
        help="Index from SLURM_ARRAY_TASK_ID; if set, process only that indexed file. If omitted, process all files."
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

    args = argparser.parse_args(argv)

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

    return args


# evaluate the entered arguments based on requirements and whether the 'slurm' flag is raised
if __name__ == "__main__":
    args = get_args()

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
        }

        if args.group:
            slurm_vars.append(f"group={args.group}")
        if args.years:
            slurm_vars.append(f"years={args.years}")

            if args.resource in array_resources:
                months = array_span_from_years(args.years)
                num_jobs = ceil(months / args.files_per_job)
                array_spec = f"0-{num_jobs - 1}"

        if args.batchsize:
            slurm_vars.append(f"batchsize={args.batchsize}")
        if args.files_per_job:
            slurm_vars.append(f"files_per_job={args.files_per_job}")

        if args.sample is not None:
            slurm_vars.append(f"sample={args.sample}")
        if args.target is not None:
            slurm_vars.append(f"target={args.target}")
        
        # Location-labeling sampling controls (forwarded to label_location)
        if getattr(args, "maxitems", None) is not None:
            slurm_vars.append(f"maxitems={args.maxitems}")
        if getattr(args, "maxfiles", None) is not None:
            slurm_vars.append(f"maxfiles={args.maxfiles}")
        if getattr(args, "maxradius", None) is not None:
            slurm_vars.append(f"maxradius={args.maxradius}")

        slurm_script = CODE_DIR / "slurm.sh"
        concurrency_cap = args.numjob  # number of simultaneous tasks
        array_flag = f"{array_spec}%{concurrency_cap}" if array_spec else None

        log_dir = PROJECT_ROOT / "slurm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        job_tag = _build_job_tag(args)
        log_token = "%A_%a" if array_spec else "%j"
        stdout_path = log_dir / f"{job_tag}__{log_token}.out"
        stderr_path = log_dir / f"{job_tag}__{log_token}.err"

        cmd_parts = [
            "sbatch",
            "--job-name", job_tag,
            "--output", str(stdout_path),
            "--error", str(stderr_path),
            "--export", f"ALL,{','.join(slurm_vars)}",
        ]
        if array_flag:
            cmd_parts.extend(["--array", array_flag])
        cmd_parts.append(str(slurm_script))

        cmd = _shell_join(cmd_parts)
        print(f"[cli] submitting: {cmd}")
        os.system(cmd)
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
        if args.sample is not None:
            cmd_parts.extend(["-c", str(args.sample)])
        if args.target is not None:
            cmd_parts.extend(["-S", args.target])
        if getattr(args, "maxitems", None) is not None:
            cmd_parts.extend(["--maxitems", str(args.maxitems)])
        if getattr(args, "maxfiles", None) is not None:
            cmd_parts.extend(["--maxfiles", str(args.maxfiles)])
        if getattr(args, "maxradius", None) is not None:
            cmd_parts.extend(["--maxradius", str(args.maxradius)])
        if args.files_per_job:
            cmd_parts.extend(["--files-per-job", str(args.files_per_job)])

        # Pretty log line only
        print("[cli] running:", subprocess.list2cmdline([str(p) for p in cmd_parts]))

        # Cross-platform execution without shell-quoting issues
        subprocess.run(cmd_parts, check=True)
