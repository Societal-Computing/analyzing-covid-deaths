import numpy as np
import pandas as pd
import preprocessor as p

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModel

from tqdm import tqdm

tqdm.pandas()

from helpers.preprocess_tweets import preprocess
from helpers.minhash import find_duplicates, remove_duplicates
from helpers.model import CovidClassifier
from helpers.train import make_predictions, get_ids_and_attn_masks

UNLABELED_DATA_FILE = "./data/unlabeled/english_filtered_sorted.tsv"
LABELED_DATA_FILE = "./out/data/english_training.csv"
RETWEET_IDS = "./data/computed/retweet_tweet_ids.txt"
TWEET_WITH_MEDIA_IDS = "./data/computed/tweets_with_media_ids.txt"

MERGED_DATA_FILE = "./out/data/combined_no_retweets_english_sorted.csv"

TRAINED_MODEL = "./out/model/model.bin"
MODEL_NAME = "digitalepidemiologylab/covid-twitter-bert-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running predictions on {DEVICE}.")

BATCH_SIZE = 32
DROPOUT = 0.1
N_CLASSES = 2

if __name__ == "__main__":
    df = pd.read_csv(UNLABELED_DATA_FILE, delimiter="\t")

    # df["processed_text"] = df["full_text"].progress_apply(p.clean).progress_apply(preprocess)
    df["processed_text"] = df["full_text"].progress_apply(preprocess)
    df["processed_text"] = df["processed_text"].apply(str.strip)

    # Load the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, config=config)

    inp_ids, attn_masks, _ = get_ids_and_attn_masks(
        tokenizer, df.processed_text, labels=None
    )

    unlabeled_set = TensorDataset(inp_ids, attn_masks)

    unlabeled_dl = DataLoader(unlabeled_set, batch_size=BATCH_SIZE)

    covid_model = CovidClassifier(
        base_model=model, n_classes=N_CLASSES, dropout=DROPOUT
    )
    covid_model.load_state_dict(torch.load(TRAINED_MODEL))
    covid_model.cuda()

    predictions = make_predictions(model=covid_model, dl=unlabeled_dl, DEVICE=DEVICE)

    predictions = list(map(lambda x: "yes_label" if x else "no_label", predictions))

    df[
        "does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author"
    ] = predictions

    # merge unlabeled and labeled dataset
    labeled_df = pd.read_csv(LABELED_DATA_FILE)

    # drop the retweets from the labeled dataset before merging
    retweets = set(map(int, open(RETWEET_IDS, "r").readlines()))

    labeled_df = labeled_df[~labeled_df.tweet_id.isin(retweets)]

    # remove the processed text column
    df = df.drop("processed_text", axis=1)

    merged_df = pd.concat([labeled_df, df])

    # drop overlapping tweets from the merged dataset
    # keeping the ones from the labeled dataset
    merged_df = merged_df.drop_duplicates(subset=["full_text"], keep="first")

    # sort the merged dataset by date
    merged_df = merged_df.sort_values(by="date", ascending=True)

    # Remove tweets that have a URL in the text but have no media attached
    #
    # First, determine the tweets that have a URL in the text
    merged_df["has_url"] = merged_df.full_text.str.contains(
        "https?:\/\/", regex=True
    ).tolist()

    # Next, determine the tweets that have media attached
    tweets_with_media_ids = set(map(int, open(TWEET_WITH_MEDIA_IDS).readlines()))

    has_media_list = []
    for tid in merged_df.tweet_id:
        if tid in tweets_with_media_ids:
            has_media_list.append(True)
        else:
            has_media_list.append(False)

    merged_df["has_media"] = has_media_list

    # Finally, remove the tweets that have a URL in the text but have no media attached
    merged_df = merged_df[~(merged_df.has_url & ~merged_df.has_media)]

    print("Merged dataset size: ", merged_df.shape)

    # Applying minhashLSH to remove any duplicate tweets still remaining
    duplicates_dict = find_duplicates(merged_df)

    merged_df = remove_duplicates(merged_df, duplicates_dict)

    print("Merged dataset size after removing duplicates: ", merged_df.shape)

    merged_df.to_csv(MERGED_DATA_FILE, index=False)
