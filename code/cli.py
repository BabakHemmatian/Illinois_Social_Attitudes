import argparse
import os
from utils import groups, validate_years, array_span_from_years
from pathlib import Path
from math import ceil

# set path variables
dir_path = os.path.dirname(os.path.realpath(__file__))  # kept for backward-compat
CODE_DIR = Path(__file__).resolve().parent              # absolute /code
PROJECT_ROOT = CODE_DIR.parent                          # absolute project root

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
        '-r', '--resource',
        type=str,
        choices=[
            'filter_keywords', 'filter_language', 'filter_sample',
            'filter_relevance', 'filter_keywords_adv','metrics_interrater', 'label_moralization',
            'label_generalization', 'label_sentiment', 'train_relevance', 'label_emotion','label_location'
        ],
        required=True,
        help="Indicate the type of processing needed. Labeling and metrics options require the output of filtering steps for the indicated years. Filtering should be done with consecutive commands in order: keywords, language, then relevance."
    )
    argparser.add_argument(
        '-g', '--group',
        type=str,
        choices=list(groups.keys()),
        required=True,
        help='Identify the social group to which the processing should be applied.'
    )
    argparser.add_argument(
        '-y', '--years',
        type=str,
        help=f'Determine the range of years to which the tool should be applied for the indicated groups. Must be either a number between 2007 and 2023 or a range in that time frame with the start and end separated by a dash.'
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
        '-j',"--num-jobs",
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

    # Validate years if required
    if args.resource in needs_years:
        if not args.years:
            argparser.error("--years is required for this resource")
        else:
            validate_years(args.years, argparser)

    # Validate batchsize if required
    if args.resource in needs_batchsize:
        if args.batchsize is None:
            argparser.error("--batchsize is required for this resource")
        elif args.batchsize <= 0:
            argparser.error("--batchsize must be a positive integer")

    return args

# evaluate the entered arguments based on requirements and whether the 'slurm' flag is raised
if __name__ == "__main__":
    args = get_args()

    if args.slurm:
        # Include type as one of the exported SLURM vars
        slurm_vars = (
            f"resource={args.resource},"
            f"group={args.group},"
            f"type={args.type}"
        )

        array_spec = None

        if args.years:
            slurm_vars += f",years={args.years}"
            months = array_span_from_years(args.years)  # total month-files
            files_per_job = args.files_per_job
            num_jobs = ceil(months / files_per_job)
            array_spec = f"0-{num_jobs-1}"

        if args.batchsize:
            slurm_vars += f",batchsize={args.batchsize}"

        if args.files_per_job:
            slurm_vars += f",files_per_job={args.files_per_job}"

        # Location-labeling sampling controls (forwarded to label_location)
        if getattr(args, "maxitems", None) is not None:
            slurm_vars += f",maxitems={args.maxitems}"
        if getattr(args, "maxfiles", None) is not None:
            slurm_vars += f",maxfiles={args.maxfiles}"
        if getattr(args, "maxradius", None) is not None:
            slurm_vars += f",maxradius={args.maxradius}"

        slurm_script = str(CODE_DIR / "slurm.sh")

        concurrency_cap = args.numjob  # number of simultaneous tasks
        array_flag = f"--array={array_spec}%{concurrency_cap}" if array_spec else ""

        cmd = f'sbatch {array_flag} --export=ALL,{slurm_vars} "{slurm_script}"'
        print(f"[cli] submitting: {cmd}")
        os.system(cmd)
    else:
        # Robust path to the resource script inside code/
        resource_script = str(CODE_DIR / f"{args.resource}.py")
        cmd = f'python "{resource_script}" -t {args.type} -r {args.resource} -g {args.group}'
        if args.years:
            cmd += f' -y {args.years}'
        if args.batchsize:
            cmd += f' -b {args.batchsize}'
        # Forward array index and location-labeling knobs when running locally
        if args.array is not None:
            cmd += f" --array {args.array}"
        if getattr(args, "maxitems", None) is not None:
            cmd += f" --maxitems {args.maxitems}"
        if getattr(args, "maxfiles", None) is not None:
            cmd += f" --maxfiles {args.maxfiles}"
        if getattr(args, "maxradius", None) is not None:
            cmd += f" --maxradius {args.maxradius}"
        if args.files_per_job:
            cmd += f" --files-per-job {args.files_per_job}"
        os.system(cmd)
