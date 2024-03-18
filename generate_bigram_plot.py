import html
import nltk
import string
import wordcloud

import numpy as np
import pandas as pd
import preprocessor as p
import matplotlib.pyplot as plt

from collections import Counter

from tqdm import tqdm
from PIL import Image, ImageChops

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from helpers.data import load_covid_death_tweets

p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)

PLOT_DIR = "./out/plots/"
COVID_DEATH_TWEETS = "./out/data/regex_filtered_english_with_death_labels.csv"


def convert_to_dict(bg, cv):
    words = cv.get_feature_names_out()
    counts = bg.sum(axis=0).T

    d = {w: c.item() for w, c in zip(words, counts)}
    return d


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


df = load_covid_death_tweets(COVID_DEATH_TWEETS)

df = df[
    df.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
    == "yes_label"
].reset_index(drop=True)

df.date = pd.to_datetime(df.date)
df = df[(df["date"] >= "2020-03-01T00:00:00") & (df["date"] <= "2021-03-31T23:59:59")]

df = df[
    df.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
    == "yes_label"
]

df = df[
    (df.country_name == "United States") | (df.country_name == "United Kingdom")
].reset_index(drop=True)

# WORDS to skip, includes the one used in the regex/keyword filtering
words_to_skip = [
    "succumbed",
    "battle",
    "lost",
    "passed",
    "away",
    "my",
    "died",
    "covid",
    "covid19",
    "19",
    "corona",
    "coronavirus",
    "virus",
    "im",
]
relationships = [
    "father",
    "dad",
    "mother",
    "mom",
    "mum",
    "pop",
    "pops",
    "grandfather",
    "grandad",
    "granddad",
    "grandmom",
    "grandmother",
    "grandpa",
    "grandma",
    "brother",
    "sister",
    "son",
    "daughter",
    "husband",
    "wife",
    "uncle",
    "aunt",
]
time_words = [
    "last",
    "week",
    "day",
    "month",
    "night",
    "morning",
    "ago",
    "weeks",
    "days",
    "nights",
    "months",
    "due",
    "today",
]

words_to_skip = (
    words_to_skip + relationships + time_words + list(stopwords.words("english"))
)


texts = df.full_text.apply(p.clean).apply(str.lower).apply(str.strip)

cv = CountVectorizer(ngram_range=(2, 2), stop_words=words_to_skip)
bigrams = cv.fit_transform(texts)

bigram_counts = convert_to_dict(bigrams, cv)

bcloud = wordcloud.WordCloud(
    width=1360, height=768, background_color="white", max_words=100
).generate_from_frequencies(bigram_counts)

fig, ax = plt.subplots(figsize=(12, 6))
plt.imshow(bcloud)
# plt.title("Bigrams from Tweets mentioning a Covid related death (classifier filtered)")

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Convert plot to CMYK image
fig.canvas.draw()

img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

img = Image.fromarray(img_arr, mode="RGB")
img = img.convert("CMYK")
img = trim(img)
img.save(
    PLOT_DIR + "bigrams.pdf",
    dpi=(300, 300),
)
