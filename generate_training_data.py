import glob
import numpy as np
import pandas as pd

import logging

logging.basicConfig(level=logging.DEBUG)

from tqdm import tqdm

# for showing progress bar for pandas functions
tqdm.pandas()

# We only look at the aggregated dataset
# aggregated dataset starts with a{job_id}.tsv
DATA_DIR = "./data/labeled/*/a*.csv"
OUTPUT_FILE = "./out/data/english_training.csv"


def get_optimal_death_label(row):
    """
    Returns the most frequent label for death
    If there's a tie, it returns the label with the highest confidence
    """
    grouper = df.groupby(
        [
            "full_text",
            "does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author",
        ]
    )[
        "does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author:confidence"
    ]

    counts = grouper.count()[row.full_text]
    counts_mean = grouper.mean()[row.full_text]

    if len(counts) == 1:
        # if there's only one label, return that
        return counts.index[0]

    if counts["yes_label"] == counts["no_label"]:
        # for tie breaker check the mean of confidence
        return counts_mean.sort_values(ascending=False).index[0]
    elif counts["yes_label"] > counts["no_label"]:
        # if yes_label is more, return that
        return "yes_label"
    else:
        return "no_label"


def get_the_optimal_relationship_label(row):
    """
    Returns the most frequent label for relationship
    If there's a tie, it returns the label with the highest confidence
    If there's no death mentioned return, nan
    """
    if (
        row.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
        == "no_label"
    ):
        return np.nan

    grouper = df.groupby(
        [
            "full_text",
            "what_is_the_relationship_between_the_tweets_author_and_the_victim_mentioned",
        ]
    )[
        "what_is_the_relationship_between_the_tweets_author_and_the_victim_mentioned:confidence"
    ]

    counts = grouper.count()[row.full_text]
    counts_mean = grouper.mean()[row.full_text]

    if len(counts) == 1:
        # if there's only one label, return that
        return counts.index[0]

    if (counts.sum() / counts)[0] == counts.values[0]:
        # for tie breaker check the mean of confidence
        return counts_mean.sort_values(ascending=False).index[0]

    # otherwise, return the most frequent label
    return counts.sort_values(ascending=False).index[0]


def get_the_optimal_relative_time_label(row):
    """
    Returns the most frequent label for relative time
    If there's a tie, it returns the label with the highest confidence
    If there's no death mentioned return, nan
    """
    if (
        row.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
        == "no_label"
    ):
        return np.nan

    grouper = df.groupby(
        [
            "full_text",
            "relative_to_the_time_of_the_tweet_when_did_the_mentioned_death_occur",
        ]
    )["relative_to_the_time_of_the_tweet_when_did_the_mentioned_death_occur:confidence"]

    counts = grouper.count()[row.full_text]
    counts_mean = grouper.mean()[row.full_text]

    if len(counts) == 1:
        # if there's only one label, return that
        return counts.index[0]

    if (counts.sum() / counts)[0] == counts.values[0]:
        # for tie breaker check the mean of confidence
        return counts_mean.sort_values(ascending=False).index[0]

    # otherwise, return the most frequent label
    return counts.sort_values(ascending=False).index[0]


def prepare_training_data(df):
    # Removing 159 tweets with no death labels
    # TODO: Review Appen
    df = df[
        ~df.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author.isna()
    ]
    logging.info(f"Removing NaN labels. Total tweets: {len(df)}")

    # Remove tweets with NaN in the full_text columns
    df = df.dropna(subset=["full_text"], axis=0)
    df = df.reset_index().drop("index", axis=1)
    logging.info(f"Removing null values in full_text. Total tweets: {len(df)}")

    # Since there are duplicate tweets (due to retweets, replies, etc.) and some of them have conflicting labels
    # for each tweet text, we select the best label based on the confidence of annotation from Appen

    # Create a copy of the dataframe
    df_temp = df.copy()

    # for the death label
    logging.info("Keeping the most confident death label.")
    df_temp[
        "does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author"
    ] = df_temp.progress_apply(get_optimal_death_label, axis=1)

    # For the relationship label
    logging.info("Keeping the most confident relationship label.")
    df_temp[
        "what_is_the_relationship_between_the_tweets_author_and_the_victim_mentioned"
    ] = df_temp.progress_apply(get_the_optimal_relationship_label, axis=1)

    # For the relative time label
    logging.info("Keeping the most confident relative time label.")
    df_temp[
        "relative_to_the_time_of_the_tweet_when_did_the_mentioned_death_occur"
    ] = df_temp.progress_apply(get_the_optimal_relative_time_label, axis=1)

    # Now, we can drop the duplicates to get the retweets/replies free tweets
    # We only keep the first occurrence of a tweet and drop the other duplicates
    df_temp.date = pd.to_datetime(df_temp.date)
    df_temp = df_temp.sort_values(by="date", ascending=True)
    df_temp = df_temp.drop_duplicates(
        subset=[
            "full_text",
            "does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author",
            "relative_to_the_time_of_the_tweet_when_did_the_mentioned_death_occur",
            "what_is_the_relationship_between_the_tweets_author_and_the_victim_mentioned",
        ],
        keep="first",
    )

    # Save the dataframe
    df_temp.to_csv(OUTPUT_FILE, index=False)
    logging.info("Dataset saved as 'english_training.csv'.")
    return


if __name__ == "__main__":
    # Load and concat the CSV from the two jobs
    df = pd.concat([pd.read_csv(job_file) for job_file in glob.glob(DATA_DIR)])

    # Sort the dataframe by date
    df = df.sort_values(by="date", ascending=True)

    logging.info(
        f"English tweets loaded and concatenated. Total tweets: {len(df)}, Unique tweet ids: {df.tweet_id.nunique()}, Unique texts: {df.full_text.nunique()}"
    )

    prepare_training_data(df)
