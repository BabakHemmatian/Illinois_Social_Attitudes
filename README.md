# Illinois_Social_Attitudes
This repository contains tools for the development and evaluation of the Illinois Social Attitudes Aggregate Corpus (ISAAC), a comprehensive dataset of Reddit discourse about social groups defined by race, skin tone, weight, sexuality, age and ability.

Once the virtual environment including all the necessary packages has been created and the correct directory structure is established for the models (see below), you can use use command line arguments with cli.py to make use of the resources. Use --help to receive more information about the available options. Example:
```
python cli.py -resource filter_keywords -group sexuality -years 2007-2009
```

# Virtual Environment Creation
TBA
# Pre-trained Models
The directory structure and the naming format for storing the models that the scripts access should be as follows:

![models_folder_structure](https://github.com/user-attachments/assets/294de346-145e-4548-bc02-86569b8d8093)

The list of models will be expanded to include all the resource options for all of the social groups included in ISAAC. 
