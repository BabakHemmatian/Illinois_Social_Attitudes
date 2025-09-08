# import helper functions, kept in a separate file for readability
from cli import get_args, dir_path
from utils import dataset_split,split_dataset_to_file,split_dataset_from_file,f1_calculator

# Import needed python packages and functions
import torch
from transformers.file_utils import is_torch_available
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import csv
import random
from sklearn.utils import compute_class_weight
from collections import Counter
from datetime import datetime
import os

# NOTE: This code is for development purposes only. Once a neural network model is considered final, the trial number is manually replaced at the end of the model folder's name with "final", and that will be the path future scripts will use for performing classification.

args = get_args()
group = args.group

### Run settings

## Set hyperparameters for training/testing
trial = 2 # so that we can run multiple trials of the same model with different hyperparameters. Change to a higher number to run a new trial. Set to "final" with training=False for testing evaluating the final model.
training = False # If training a new model, set this variable to True. If reloading from disk, set it to False. 
max_length = 512 # Number of tokens allowed in a single document. Tokens further than 512 are unlikely to change the category, so the classifier will not scan them.
train_batch_size = 8 # Batch size for training. If you have a lot of RAM (or GPU RAM, if you are running the script on a GPU), you can increase it.
thresholding = True # If True, the model will use a confidence threshold (set below) to determine the class of a document. If False, it will always return the most probable class.
threshold_class = 1 # the class that needs a probability passing the threshold (set below) to be picked as the answer. Only matters if thresholding = True.
threshold = 0.6 # The confidence threshold for the rarest class. If the model's confidence in a class is below this value, it will not return that class. Only matters if thresholding=True and the value is greater than >.50 given the two main labels. 
epochs = 1 # Number of times the model weights are adjusted by going through the entire training set during training.
custom_training = False # If True, the model will use a custom training loop that penalizes certain types of mistakes more heavily. If False, it will use standard training.
penalty_weight = 1 # Only matters if custom_training is True. adjust as needed (e.g., 0.5, 1.0, 2.0)

## choose the model to use

# NOTE: RoBERTa from 2019 is a basic, but good choice for many general use cases
# NOTE: roberta-base is a smaller model, roberta-large is a larger model with more parameters. RoBERTa-large was overfitting to training noise, so I used the smaller version.
model_name = "roberta-base" # or "roberta-large" for a larger model

## Sets path variables
model_path = dir_path.replace("code","models\\filter_relevance_{}_{}_{}".format(group,model_name,trial))
ratings_path = os.path.join(dir_path.replace("code","data\\data_relevance_ratings\\"))

## Define a function to set a random seed for repeatable results in subsequent runs
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy`` and ``torch``, if installed.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

set_seed(1) # set a random seed for reproducibility

### Create the object that breaks docs into BERT tokens (i.e., subwords)
tokenizer = RobertaTokenizerFast.from_pretrained(model_name,do_lower_case=True)

### Prepare the human-rated offloading data

# NOTE: the code below assumes two primary annotators, with a tie-breaker third
# NOTE: there are very few "Other (write comment)" labels. Not enough to train a good classifier for it. I'm excluding it from training.
# There is no perfect solution to the problem above, but you can ask me about alternative approaches (e.g., weighting, oversampling and synthetic data).

## Mapping between the labels and their titles
title_label = {"Irrelevant":0, "Relevant":1}
label_title = {0:"Irrelevant", 1:"Relevant"}
num_labels = len(label_title)

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
                if line[2].strip() == "x":
                    ratings[rater][int(line[0].strip())] = 0
                else:
                    ratings[rater][int(line[0].strip())] = int(line[2].strip())

# Consider a post as relevant if at least one rater has marked it as relevant
final_ratings = {}
for id_ in ratings[0]:
    if ratings[0][id_] == 0 and ratings[1][id_] == 0:
        final_ratings[id_] = 0
    else:
        final_ratings[id_] = 1

## create the final list of documents and labels based on which the model will be trained and evaluated
final_labels = list(final_ratings.values())
texts = list(texts.values())

print("Number of annotated docs used in training: {}".format(len(texts)))
print(f"Number of instances for each label: {Counter(final_labels)}") # print the count of each label in the dataset

## split the dataset into training/validation/test sets if no prior split is found on file, load the prior split otherwise

# the list of files needed for recreating a previous training/validation/test split (retained for reproducibility)
split_data_path = model_path.replace("filter_relevance_{}_{}_{}".format(group,model_name,trial),"train_relevance_data_split")
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

if missing_file: # if a file is missing

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

else: # if all files are accounted for

    print("Loading predetermined training, validation and test sets (80/10/10 split)")
    train_texts = split_dataset_from_file(file_list[0])
    train_labels = split_dataset_from_file(file_list[1])
    valid_texts = split_dataset_from_file(file_list[2])
    valid_labels = split_dataset_from_file(file_list[3])
    test_texts = split_dataset_from_file(file_list[4])
    test_labels = split_dataset_from_file(file_list[5])

## Prepare training, validation and test data for training and model evaluation

print("Number of training documents: {}".format(len(train_texts)))
print(f"Number of instances for each label in training data: {Counter(train_labels)}") 
print("Number of validation documents: {}".format(len(valid_texts)))
print(f"Number of instances for each label in validation data: {Counter(valid_labels)}") 
print("Number of test documents: {}".format(len(test_texts)))
print(f"Number of instances for each label in validation data: {Counter(test_labels)}") 

# Transform the labels from a simple list. The transformers library requires labels to be in LongTensor, not Int format
train_labels=torch.from_numpy(np.array(train_labels)).type(torch.LongTensor)
valid_labels=torch.from_numpy(np.array(valid_labels)).type(torch.LongTensor)
test_labels=torch.from_numpy(np.array(test_labels)).type(torch.LongTensor)

## Assign different weights to the labels so that rarer ones are nonetheless prioritized during training
weights = list(compute_class_weight('balanced', classes=np.asarray(np.unique(train_labels)), y=np.asarray(train_labels)))
weights[1] = weights[1] * .6 # increase or decrease the weight of the "Relevant" class to more strongly account for rarity or over-emphasis

print(f"Class weights to account for imbalanced training data: {weights}")

## Tokenize the training and validation data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

## Transformers requires custom datasets to be transformed into PyTorch datasets before training. The following function makes the transition
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

### train or load the model

# NOTE: If using NVIDIA GPUs, this would be where CUDA would be invoked for faster processing. 
# NOTE: The code would automatically use CPU if no GPU devices are set up for the task.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Performing computations on {device}") 

## Set training arguments
# NOTE: If the performance is not good, adjusting the training batch size would be most helpful.

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=epochs,              # total number of training epochs
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=20,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=20,               # log & save weights each logging_steps
    save_steps=20,
    eval_strategy="steps",     # evaluate each `logging_steps`
)
    
## Define a "WeightedTrainer" class so that we can use the imbalanced label weights defined above

if not custom_training: # if simply aligning weights with rarity in training data
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 2 labels with different weights)
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([float(i) for i in weights]).to(device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
else: # if using a trainer that penalizes when model predicts 0 but label is 1
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, penalty_weight=penalty_weight):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # Standard weighted cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([float(i) for i in weights]).to(logits.device))
            ce_loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

            # Predictions
            preds = torch.argmax(logits, dim=-1)

            # Confusion penalty: penalize when model predicts 0 but label is 1
            confusion_mask_1_to_0 = (labels == 1) & (preds == 0)

            penalty = penalty_weight * confusion_mask_1_to_0.sum().float()

            # Final loss
            total_loss = ce_loss + penalty

            return (total_loss, outputs) if return_outputs else total_loss

## Train or load the model as appropriate

if training: # train a new model and save it

    print(f"Training a new classifier for the relevance of posts to the {group} social group...")

    # Load the model from the Hugging Face model hub
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    # NOTE: num_labels is the number of classes in the classification task, which is 3 in this case (0, 1, 2)
            
    # Create the trainer object with variables defined so far
    trainer = WeightedTrainer(
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

else: # load a trained model from disk for evaluation if not training

    print(f"Loading a pretrained classifier for the relevance of posts to the {group} social group...")

    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)

### Evaluation

# defines a function that predicts the label for a given input text.
# NOTE: Can apply a confidence threshold for picking a particular class.

def get_prediction(text, threshold_class=threshold_class, threshold=threshold, thresholding=thresholding): 
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # Model inference
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)[0]  # (2,) for binary classification

    # Apply thresholding: only allow class 0 if confident enough
    if thresholding and probs[threshold_class] > threshold:
        return threshold_class
    else:
        if thresholding:
            # Mask out class 0 if it's not confident enough
            masked_probs = probs.clone()
            masked_probs[threshold_class] = -1
            return masked_probs.argmax().item()
        else:
            return probs.argmax().item()

## Get the labels for each of the test set documents and write the results of the test to disk

predictions = [get_prediction(text) for text in test_texts]
with open("{}/test_results_{}_{}_{}.csv".format(model_path,group,model_name,trial),"w",encoding='utf-8',errors='ignore',newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text","true_label","predicted_label"])
    for text,true_label,predicted_label in zip(test_texts,test_labels,predictions):
        writer.writerow([text,label_title[int(true_label)],label_title[int(predicted_label)]])

## Evaluate performance on the test set

#NOTE: Since the sklearn functions for precision, recall and f1 did not work properly, I wrote a short script in utils.py that calculates them from scratch
precision,recall,f1 = f1_calculator([int(label) for label in test_labels],[int(pred) for pred in predictions])

# Save evaluation performance 
with open("{}/test_results_{}_{}_{}.txt".format(model_path,group,model_name,trial),"a+",encoding='utf-8',errors='ignore') as f:
    f.write("***{}***\n".format(datetime.now()))
    f.write("training batch size: {}\n".format(train_batch_size))
    f.write("class weights: {}\n".format(weights))
    f.write("custom training: {}\n".format(custom_training))
    f.write("penalty weight: {}\n".format(penalty_weight))
    f.write("thresholding: {}\n".format(thresholding))
    f.write("threshold class: {}\n".format(threshold_class))
    f.write("threshold: {}\n".format(threshold))
    f.write("Number of epochs: {}\n".format(epochs))
    f.write("Performance on the test set:\n")
    f.write("Precision: {}\n".format(precision))
    f.write("Recall: {}\n".format(recall))
    f.write("F1: {}\n".format(f1))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
