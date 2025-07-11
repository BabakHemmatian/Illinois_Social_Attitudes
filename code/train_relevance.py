# import base directory path
from cli import dir_path
from utils import dataset_split,split_dataset_to_file,split_dataset_from_file,f1_calculator

# Import needed python packages and functions
import torch
from transformers.file_utils import is_torch_available
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import csv
import random
from sklearn import utils
import math
from collections import Counter
import pathlib
import os
import sys

# NOTE: This code is for development purposes only. Once a neural network model is considered final, 
# the trial number is manually removed from the end of the model folder's name, and that will be the 
# address which filter_relevance.py uses for data pruning.

# task outline
group = 'weight'
trial = 0
training = True # If training a new model, set the variable below to True. If loading from disk, set it to False

# input and output paths
ratings_path = os.path.join(dir_path.replace("code","data\\data_relevance_ratings\\"))
model_path = dir_path.replace("code","models\\filter_relevance_{}_roberta_large_{}".format(group,trial))

# Set random seed for repeatable results in subsequent runs
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

set_seed(1)

# choose the model to use
# NOTE: RoBERTa from 2019 is a basic, but good choice for many use cases
model_name = "roberta-large"

max_length = 512 # Number of tokens allowed in a single document
# NOTE: Tokens further than 512 are unlikely to change whether a text is relevant or not

# Create the object that breaks docs into BERT tokens
tokenizer = RobertaTokenizerFast.from_pretrained(model_name,do_lower_case=True)

### Prepare the relevance rating data

# NOTE: the code below assumes two annotators
texts = {} 
ratings = {0:{},1:{}}
for rater in range(2):
    with open(ratings_path+"relevance_sample_{}_{}_rated.csv".format(group,rater),"r", encoding='utf-8',errors='ignore') as f:
        reader = csv.reader(f)
        for idx,line in enumerate(reader):
            if idx != 0 and len(line) > 0:
                if rater == 0:
                    texts[int(line[0].strip())] = line[1].strip()
                ratings[rater][int(line[0].strip())] = int(line[2].strip())

# resolve disagreements between annotators
labels = {}
for id_ in ratings[0]:
    if ratings[0][id_] == ratings[1][id_]:
        if ratings[0][id_] == -1:
            labels[id_] = 0
        else:
            labels[id_] = ratings[0][id_]
    elif ratings[0][id_] == 1 or ratings[1][id_] == 1:
        labels[id_] = 1
    else:
        labels[id_] = 0

# create the final list of documents based on which the model will be trained and evaluated
texts = list(texts.values())
final_labels = list(labels.values())
print(" Number of annotated comments: {}".format(len(texts)))
print(Counter(final_labels)) # print the number of positive and negative examples in the data

# the list of files needed for recreating a previous training/validation/test split
split_data_path = model_path.replace("filter_relevance_{}_roberta_large_{}".format(group,trial),"train_relevance_data_split")
split_data = ["training","validation","test"]
file_list = []
for cat in split_data:
    file_list.append(f"{split_data_path}\\text_{group}_{cat}.csv")
    file_list.append(f"{split_data_path}\\label_{group}_{cat}.txt")

# see if any of the needed data split files are missing
missing_file = 0
for file in file_list:
    if not os.path.exists(file):
        missing_file = 1

# split the dataset into training/validation/test sets if no prior split is found on file, load the prior split otherwise
if missing_file:

    print("Creating training, validation and test sets (80/10/10 split)")

    aligned_files = {}

    train_texts, valid_texts_init, train_labels, valid_labels_init = dataset_split(texts,final_labels,proportion=0.8)
    valid_texts, test_texts, valid_labels, test_labels = dataset_split(valid_texts_init,valid_labels_init,proportion=0.5)
    
    split_dataset_to_file(file_list[0],train_texts)
    split_dataset_to_file(file_list[1],train_labels)
    split_dataset_to_file(file_list[2],valid_texts)
    split_dataset_to_file(file_list[3],valid_labels)
    split_dataset_to_file(file_list[4],test_texts)
    split_dataset_to_file(file_list[5],test_labels)

else:

    print("Loading predetermined training, validation and test sets (80/10/10 split)")
    train_texts = split_dataset_from_file(file_list[0])
    train_labels = split_dataset_from_file(file_list[1])
    valid_texts = split_dataset_from_file(file_list[2])
    valid_labels = split_dataset_from_file(file_list[3])
    test_texts = split_dataset_from_file(file_list[4])
    test_labels = split_dataset_from_file(file_list[5])

# Transform the labels from a simple list. The transformers library requires labels to be in LongTensor, not Int format
train_labels=torch.from_numpy(np.array(train_labels)).type(torch.LongTensor)#data type is long
valid_labels=torch.from_numpy(np.array(valid_labels)).type(torch.LongTensor)#data type is long
test_labels=torch.from_numpy(np.array(test_labels)).type(torch.LongTensor)#data type is long

print("Number of training documents: {}".format(len(train_texts)))
print("Number of validation documents: {}".format(len(valid_texts)))
print("Number of test documents: {}".format(len(test_texts)))

weights = list(utils.compute_class_weight('balanced', classes=np.array(np.unique(train_labels)), y=np.array(train_labels)))
print(f"Class weights to account for imbalanced training data: {weights}")

# Tokenize the training and validation data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

# Transformers requires custom datasets to be transformed into PyTorch datasets before training. The following function makes the transition
class relevance_data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = relevance_data(train_encodings, train_labels)
valid_dataset = relevance_data(valid_encodings, valid_labels)
test_dataset = relevance_data(test_encodings, test_labels)

### train or load the model

# NOTE: If we were using GPUs, this would be where CUDA would be invoked

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Performing computations on {device}")
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=400,               # log & save weights each logging_steps
    save_steps=400,
    eval_strategy="steps",     # evaluate each `logging_steps`
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([float(i) for i in weights]).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

if training: # train a new model and save it

    print(f"Training a new classifier for the relevance of posts to the {group} social group...")
    
    # Create the trainer object with variables defined so far
    trainer = CustomTrainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
    )

    # Train the model
    trainer.train()

    # saving the fine-tuned model & tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

else: # or load a trained model from disk for evaluation

    print(f"Loading a pretrained classifier for the relevance of posts to the {group} social group...")

    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

### Evaluation

# Get the labels for each of the evaluation set documents
def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return [0,1][probs.argmax()] # 0 is irrelevant, 1 is relevant

# Since the sklearn functions for precision, recall and f1 did not work properly, I wrote a short script below that calculates them from scratch
predictions = [get_prediction(text) for text in test_texts]
precision,recall,f1 = f1_calculator(test_labels,predictions)
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1: {}".format(f1))
