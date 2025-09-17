import argparse
import os
from utils import groups, validate_years, array_span_from_years
from pathlib import Path

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
        'filter_sample',
        'label_moralization',
        'label_sentiment',
        'label_generalization',
        'label_emotion'
    ]

    # Conditionally require --batchsize
    needs_batchsize = [
        'filter_relevance',
        'label_moralization',
        'label_generalization',
        'label_emotion',
        'label_sentiment'
    ]
    
    argparser.add_argument(
        '-r', '--resource',
        type=str,
        choices=[
            'filter_keywords', 'filter_language', 'filter_sample',
            'filter_relevance', 'metrics_interrater', 'label_moralization',
            'label_generalization', 'label_sentiment', 'train_relevance', 'label_emotion'
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
    "--array",
    type=int,
    help="Index from SLURM_ARRAY_TASK_ID; if set, process only that indexed file. If omitted, process all files."
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
        slurm_vars = f"resource={args.resource},group={args.group}"
        array_spec = None

        if args.years:
            slurm_vars += f",years={args.years}"
            months = array_span_from_years(args.years)
            array_spec = f"0-{months-1}"           # e.g., 2016 -> 0-11; 2007-2008 -> 0-23

        if args.batchsize:
            slurm_vars += f",batchsize={args.batchsize}"

        slurm_script = str(CODE_DIR / "slurm.sh")

        concurrency_cap = 5 # number of simulaneous tasks
        array_flag = f"--array={array_spec}%{concurrency_cap}" if array_spec else ""

        cmd = f'sbatch {array_flag} --export=ALL,{slurm_vars} "{slurm_script}"'
        print(f"[cli] submitting: {cmd}")
        os.system(cmd)
    else:
        # Robust path to the resource script inside code/
        resource_script = str(CODE_DIR / f"{args.resource}.py")
        cmd = f'python "{resource_script}" -r {args.resource} -g {args.group}'
        if args.years:
            cmd += f' -y {args.years}'
        if args.batchsize:
            cmd += f' -b {args.batchsize}'
        os.system(cmd)
