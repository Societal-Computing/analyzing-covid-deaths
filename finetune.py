import random
import pickle

import numpy as np
import pandas as pd
import preprocessor as p
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel

from tqdm import tqdm

# configure tqdm progress bar for pandas
tqdm.pandas()

# Load the preprocessing function from the CovidBertv2 repo
from helpers.preprocess_tweets import preprocess

# Load the helper functions
from helpers.model import CovidClassifier
from helpers.train import train, evaluate_test, get_ids_and_attn_masks

# Seed for reproducing the experiment
random_seed = 42

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
transformers.set_seed(random_seed)

# Define the training data
DATA_FILE = "./out/data/english_training.csv"

# We will fine tune the CT-BERT model
# see: https://github.com/digitalepidemiologylab/covid-twitter-bert
MODEL_NAME = "digitalepidemiologylab/covid-twitter-bert-v2"


# Hyperparameters configuration for training
# based on the best run
N_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 3e-6
NUM_EPOCHS = 2
DROPOUT = 0.1

# Define the device to be used for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Trainig on: ", DEVICE)

# Preprocess the tweets
# p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.RESERVED)

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)

    # rename the columns for ease
    df[
        "death"
    ] = (
        df.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
    )

    # map the death labels: if there's a personally related death then 1 otherwise 0
    df.death = df.death.progress_apply(lambda x: 1 if x == "yes_label" else 0)

    # df["processed_text"] = df.full_text.progress_apply(p.clean).progress_apply(preprocess)
    df["processed_text"] = df.full_text.progress_apply(preprocess)

    # remove any duplicates
    df = df.drop_duplicates(subset=["processed_text", "death"])
    print("Training on: ", df.shape)

    # strip any blank spaces
    df.processed_text = df.full_text.progress_apply(str.strip)

    # Create train/val/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        df.processed_text,
        df.death,
        test_size=0.05,
        stratify=df.death,
        random_state=random_seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.02, stratify=y_train, random_state=random_seed
    )

    # Save the test set for further evaluation
    pickle.dump({"X": X_test, "y": y_test}, open("out/data/test_set.pkl", "wb"))

    # Load the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, config=config)

    # Convert the data into Pytorch datasets
    train_set = TensorDataset(
        *get_ids_and_attn_masks(tokenizer=tokenizer, sentences=X_train, labels=y_train)
    )
    val_set = TensorDataset(
        *get_ids_and_attn_masks(tokenizer=tokenizer, sentences=X_val, labels=y_val)
    )
    test_set = TensorDataset(
        *get_ids_and_attn_masks(tokenizer=tokenizer, sentences=X_test, labels=y_test)
    )

    # Convert the pytorch datasets to dataloaders for training
    train_dl = DataLoader(
        train_set, sampler=RandomSampler(train_set), batch_size=BATCH_SIZE
    )
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Define the classifier
    covid_model = CovidClassifier(
        base_model=model, n_classes=N_CLASSES, dropout=DROPOUT
    )
    _ = covid_model.cuda()

    # Run the training function
    covid_model = train(
        model=covid_model,
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=NUM_EPOCHS,
        n_classes=N_CLASSES,
        learning_rate=LEARNING_RATE,
        DEVICE=DEVICE,
    )

    # Evaluate the model
    preds, labels, avg_f1, f1_std = evaluate_test(
        model=covid_model, test_dl=test_dl, n_classes=N_CLASSES, DEVICE=DEVICE
    )
