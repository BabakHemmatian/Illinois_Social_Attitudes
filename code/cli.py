import argparse
import os
from utils import groups

dir_path = os.path.dirname(os.path.realpath(__file__))
raw_data = dir_path.replace("code","data\\data_reddit_raw\\reddit_comments")

def get_args(argv=None):
    argparser = argparse.ArgumentParser(description='A command line interface for Illinois Social Attitudes Aggregate Corpus development and evaluation functions. See the GitHub repository\'s readme file for more details on the available resources.')
    argparser.add_argument('-r','--resource', type=str, choices=['filter_keywords','filter_language','filter_sample','filter_relevance','label_moralization','label_generalization','label_sentiment','metrics_interrater'], required=True, help="Indicate the type of processing needed. Labeling and metrics options require the output of filtering steps for the indicated years. Filtering should be done with consecutive commands in order: keywords, language, then relevance.")
    argparser.add_argument('-g','--group', type=str, choices=list(groups.keys()),required=True,help='Identify the social group to which the processing should be applied.')
    argparser.add_argument('-y','--years',type=str,required=True,help='Determine the range of years to which the tool should be applied for the indicated groups. Must be either a number between 2007 and 2023 or a range with the start and end separated by a dash.')
    argparser.add_argument('-s','--slurm',action="store_true",help="Submit a Slurm job. Best used for NN resources (filter_relevance, label_moralization, label_generalization). Should only be used on a Slurm computing cluster.")
    args = argparser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()
    
    if args.slurm:
        os.system('sbatch --export=ALL,resource={},group={},years={} slurm.sh'.format(args.resource, args.group,args.years))
    else:
        os.system('python {}.py -r {} -g {} -y {}'.format(args.resource,args.resource, args.group,args.years))

# TODO: Add a .gitignore file that excludes .csv, error report and model files from commit updates. 
