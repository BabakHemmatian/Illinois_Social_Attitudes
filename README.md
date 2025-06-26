# Illinois Social Attitudes Aggregate Corpus (ISAAC)
This repository contains tools for the development and evaluation of the **Illinois Social Attitudes Aggregate Corpus (ISAAC)**, a comprehensive dataset of Reddit discourse from 2007 to 2023 about social groups defined by race, skin tone, weight, sexuality, age and ability. 

The resources allow filtering Reddit content by keywords and the use of English language. 

Pre-trained neural networks allow the automatic pruning of irrelevant content related to the social groups mentioned above. 

Further resources generate moralization and generalized language labels for the relevance-filtered datasets.

**Note:** The scripts were developed and tested on Windows 11. Cross-platform compatibility is not guaranteed.

**Note:** filter_relevance for skin tone, as well as label_sentiment and label_localization resources are in development. 

## Citation
If you use this repository in your work, please cite us as follows:
```
**APA: **
Hemmatian, B., Hadjarab, S., Yu, R. (2025). Illinois_Social_Attitudes [Computer software]. GitHub. [https://github.com/BabakHemmatian/Illinois_Social_Attitudes](https://github.com/BabakHemmatian/Illinois_Social_Attitudes)

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

Once finished, navigate to ```Illinois_Social_attitudes``` on the command line and enter ```conda create --name ISAAC --file req.txt```. Answer 'y' to the question. When finished, run ```conda activate ISAAC```.

### Commands
You can now use command line arguments with ```cli.py``` in the ```scripts``` folder to make use of the resources. Use --help to receive more information about the available options. Example:
```
python cli.py --resource filter_keywords --group sexuality --years 2007-2009
```
**Note: **"filter_relevance", "label_moralization" and "label_generalization" resources are LLM-based and would become much faster with Cuda-enabled GPU acceleration (available on Nvidia graphics cards). If you plan to use this feature, follow the steps [here](https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef) to install PyTorch with Cuda support within your new conda environment.

The scripts may be used without any changes to recreate the ISAAC corpus. To adapt them for developing new datasets, see the section below. 

## Adaptations

### Adjusting Social Groups and Related Keywords
The current list of social groups and their binary subgroups are found in ```scripts/utils.py```. 

To search the entirety of Reddit for posts potentially relevant to your dimension of interest beyond those listed, change the entries of the ```utils.py``` list and the corresponding ```--group``` argument in ```cli.py```. Then, add correctly formatted and named text files to the ```keywords``` folder that contain words helpful for identifying potentially relevant content for your use case. See the existing files for examples to follow. This code base uses ```pyahocorasick``` for extremely fast recognition of dozens of keywords in billions of posts. This package allows only alphanumeric and punctuation characters. Choose your keyword format accordingly.

### Training New Relevance Classifiers
The ```filter_sample``` resource can be used to extract stratified samples from keyword- and language-filtered documents to be annotated for the training of new relevance classifiers. The script assumes two annotators and by default generates 1500 documents per rater equally distributed across the indicated years. 

Use the ```metrics_interrater``` resource with the correct ```--group``` argument. No ```years``` argument is needed for this resource.

Once sufficient interrater agreement is reached, use the ```train_relevance``` resource to train new relevance filtering neural networks. Adjust the social ```group``` argument to your target. No ```years``` argument is needed.
