# Illinois Social Attitudes Aggregate Corpus (ISAAC)

This repository contains tools for the development and evaluation of the **Illinois Social Attitudes Aggregate Corpus (ISAAC)**, a comprehensive dataset of Reddit discourse from 2007 to 2023 about social groups defined by the following distinctions:
1. sexuality (gay-straight; 92,979,067 comments)
2. race (Black-White; 184,684,200 comments)
3. age (young-old; 330,825,613)
4. ability (abled-disabled; 23,515,610)
5. weight (fat-thin; 21,344,219)
6. skin_tone (dark-light; 67,038,276)

**Total Corpus Size: 720,386,985 Posts**

Submissions and comments in ISAAC have all been labeled for **a variety of social-psychological variables** of interest, including moralization, sentiment, generalizations and emotions. 

## Corpus Access

You can read about the list of variables included in the corpus and their definitions [here](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/variable_list.md). The data can be found [here](https://drive.google.com/drive/u/0/folders/15luTEiHnt8BnnHjYmQnS__DSUt3lODjr). 

## The Current Repository

This repository contains the scripts that allow you to rebuild ISAAC from scratch by:
- Filtering Reddit content by keywords and the use of English language. 
- Applying pre-trained neural networks to prune irrelevant content picked up by keywords (e.g., "Black" in a context other than race). 
- Generating generalized language (e.g., genericity), moralization, sentiment and emotion labels for the relevance-filtered extracted texts.

The scripts were designed to be easily adapted for developing other Reddit corpora. See the **Adaptations** section below for more information.

**Note:**
- The scripts were developed on Windows 11, then tested on Ubuntu. However, cross-platform compatibility is not guaranteed.
- The ```label_localization``` resource, identifying the country and the state where a user likely resides, is in development.

## Citation
If you use this repository in your work, please cite us as follows:

### APA Format
```
Hemmatian, B., Hadjarab, S., Yu, R. (2025). The Illinois Social Attitudes Aggregate Corpus [Computer software]. GitHub. [https://github.com/BabakHemmatian/Illinois_Social_Attitudes](https://github.com/BabakHemmatian/Illinois_Social_Attitudes)
```
### BibTex Format
```
**BibTex: **
@misc{Hemmatian2025,
  author       = {Hemmatian, Babak and Hadjarab, Sarah and Yu, Rui},
  title        = {Illinois_Social_Attitudes},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/yourusername/your-repository](https://github.com/BabakHemmatian/Illinois_Social_Attitudes)}},
}
```
## How To Use

### Repository Setup
Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) on your computer. When finished, open a command line terminal, navigate to where you would like to place the repository, then enter ```git clone https://github.com/BabakHemmatian/Illinois_Social_Attitudes.git```. Note that the raw and processed data files for the full 2007-2023 take several terabytes of space. Choose the repository location according to your use case's storage needs.

Download [this folder](https://drive.google.com/drive/folders/1TqxjRRMZ3LTGWRCMkK6_tnIo_Zg1vms1?usp=sharing) into the newly created ```Illinois_Social_Attitudes``` folder.

The raw Reddit data that the ```filter_keywords``` resource requires can be found and downloaded [here](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13). The functions currently assume Reddit Comments as the type of data, with the relevant .zst files for a given timeframe to be placed in ```data/data_reddit_raw/reddit_comments/```. 

### Virtual Environment Setup
Follow the steps [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install the desired version of Anaconda. 

Once finished, navigate to ```Illinois_Social_attitudes``` on the command line and enter ```conda create --name ISAAC python=3.12 pip```. Answer 'y' to the question. When finished, run ```conda activate ISAAC```. Once the environment is activated, run the following command to install the necessary packages: ```pip install -r req.txt```. 

### Commands
You can now use command line arguments to make use of the resources. Use ```--help``` to receive more information about the available options. 

**Example:**
```
python ./code/cli.py --resource filter_keywords --group sexuality --years 2007-2009
```
This example command will use the appropriate keyword lists from this repository to identify documents in the complete Pushshift dataset that are potentially related to sexuality, and which come from 2007-2009. 

**Note:** ```filter_relevance```, ```label_moralization```, ```label_sentiment```, ```label_generalization``` and ```label_emotion``` resources are LLM-based and would become much faster with Cuda-enabled GPU acceleration (available on Nvidia graphics cards, with a corresponding tool for Mac users). If you plan to use this feature, follow the steps [here](https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef) to install PyTorch with Cuda support within your new conda environment. You can speed up processing even further by using batch processing in a computing cluster by adding the ```--slurm``` or ```-s``` flag to your command. Note that the specific sbatch arguments in ```slurm.sh``` need to be adjusted based on the particular cluster you are using. 

**Note:** The neural resources mentioned above require the batch size argument (```-batchsize [integer]``` or ```-b [integer]```). Set it based on your RAM and GPU RAM capacity. Values between 1200 and 2500 were used during the development of ISAAC. 

**Note:** The scripts may be used without any changes to recreate the ISAAC corpus. For that purpose, the code base currently assumes the following order in the use of resources for a given social group and year range:

1. ```filter_keywords```
2. ```filter_language```
3. ```filter_relevance```
4. ```label_moralization```
5. ```label_sentiment```
6. ```label_generalization```
7. ```label_emotion```

**Note:** If you are using ```label_sentiment``` for the first time, perform the following steps before running the script: Enter ```python -m spacy download en_core_web_sm``` in the command terminal and run ```stanza.download('en')``` after importing ```stanza``` in a Python interpreter. Only then use ```cli.py``` to call the resource. 

If you plan instead to adapt the code for developing new datasets, see the section below. 

## Adaptations

### Adjusting Social Groups and Related Keywords
The current list of social groups and their binary subgroups are found in ```scripts/utils.py```. 

To search the entirety of Reddit for posts potentially relevant to your dimension of interest beyond those listed, change the key-value pairs for the groups variable in ```utils.py``` and the choices for the corresponding ```--group``` argument in ```cli.py```. Then, add correctly formatted and named text files to the ```keywords``` folder that contain words helpful for identifying potentially relevant content for your use case. See the existing files for examples to follow. This code base uses ```pyahocorasick``` for extremely fast recognition of dozens of keywords in billions of posts. This package allows only alphanumeric and punctuation characters. Choose your keyword format accordingly. Note that the resource scripts can be easily adjusted to allow for different orders of resource use or the application of new classification models (see below). 

### Training New Relevance Classifiers
The ```filter_sample``` resource can be used to extract stratified samples from keyword- and language-filtered documents to be annotated for the training of new relevance classifiers. The script assumes two annotators and by default generates 1500 documents per rater equally distributed across the indicated years. 

Use the ```metrics_interrater``` resource with the correct ```--group``` argument to evaluate interrater agreement. No ```years``` argument is needed for this resource.

Once sufficient interrater agreement is reached, use the ```train_relevance``` resource to train new relevance filtering neural networks. Adjust the social ```group``` argument to your target and change the training hyperparameters as needed. No ```years``` argument is required for this resource.
