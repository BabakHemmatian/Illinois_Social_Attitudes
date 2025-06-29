import argparse
import os
from utils import groups,validate_years

# set path variables
dir_path = os.path.dirname(os.path.realpath(__file__))
raw_data = dir_path.replace("code","data\\data_reddit_raw\\reddit_comments")

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
        'filter_sample'
        'label_moralization',
        'label_generalization'
    ]
    
    argparser.add_argument(
        '-r', '--resource',
        type=str,
        choices=[
            'filter_keywords', 'filter_language', 'filter_sample',
            'filter_relevance', 'metrics_interrater', 'label_moralization',
            'label_generalization','train_relevance'
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
        help=f'Determine the range of years to which the tool should be applied for the indicated groups. Must be either a number between 2007 and 2023 or a range in that time frame with the start and end separated by a dash. Needed for {needs_years}'
    )
    argparser.add_argument(
        '-s', '--slurm',
        action="store_true",
        help="Submit a Slurm job. Best used for NN resources (filter_relevance, label_moralization, label_generalization). Should only be used on a Slurm computing cluster."
    )

    args = argparser.parse_args(argv)

    if args.resource in needs_years:
        if not args.years:
            argparser.error(f"--years is required when --resource is one of {needs_years}")
        else:
            validate_years(args.years, argparser)

    return args

# evaluate the entered arguments based on requirements and whether the 'slurm' flag is raised
if __name__ == "__main__":
    args = get_args()

    if args.slurm:
        # Construct Slurm export variables conditionally
        slurm_vars = f"resource={args.resource},group={args.group}"
        if args.years:
            slurm_vars += f",years={args.years}"
        os.system(f'sbatch --export=ALL,{slurm_vars} slurm.sh')
    else:
        # Construct CLI call conditionally
        cmd = f'python {args.resource}.py -r {args.resource} -g {args.group}'
        if args.years:
            cmd += f' -y {args.years}'
        os.system(cmd)
