import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.stats import spearmanr, pearsonr

from helpers.data import load_covid_death_tweets


def trim(im):
    # helper for trimming the plots
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


CORRELATIONS_PLOT_DIR = "./out/plots/correlations/"
COVID_DEATH_TWEETS = "./out/data/regex_filtered_english_with_death_labels.csv"

gender_series_plot_dir = CORRELATIONS_PLOT_DIR + "gender_series/"
if not os.path.isdir(gender_series_plot_dir):
    os.mkdir(gender_series_plot_dir)

df = load_covid_death_tweets(COVID_DEATH_TWEETS)

df = df[
    df.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
    == "yes_label"
]

df_with_gender = df[~df.regex_gender.isna()].reset_index(drop=True)

df_with_gender.date = pd.to_datetime(df_with_gender.date)
df_with_gender = df_with_gender.set_index(df_with_gender.date)

# For the US
freq = "W-SAT"
shift_period = 0
date_range = pd.date_range(start="2020-04-01", end="2021-03-31", freq=freq)

country = "United States"
weekly = True

male_counts = df_with_gender[
    (df_with_gender.regex_gender == "male") & (df_with_gender.country_name == country)
]
female_counts = df_with_gender[
    (df_with_gender.regex_gender == "female") & (df_with_gender.country_name == country)
]

male_grouped_counts = male_counts.resample(freq).tweet_id.count()
female_grouped_counts = female_counts.resample(freq).tweet_id.count()

male_grouped_counts = male_grouped_counts.reindex(date_range, fill_value=0)
female_grouped_counts = female_grouped_counts.reindex(date_range, fill_value=0)

# Fixing date lag
male_grouped_counts = male_grouped_counts.shift(periods=shift_period, freq=freq)
female_grouped_counts = female_grouped_counts.shift(periods=shift_period, freq=freq)

# load CDC
cdc = pd.read_csv(
    "./data/official/Provisional_COVID-19_Deaths_by_Week__Sex__and_Age.csv"
)
cdc = cdc[cdc.State == "United States"].reset_index(drop=True)
cdc = cdc[cdc["Age Group"] == "All Ages"].reset_index(drop=True)

key = "End Week" if weekly else "End Month"
cdc[key] = pd.to_datetime(cdc[key])
cdc = cdc.set_index(cdc[key])

male_official_us = (
    cdc.groupby([pd.Grouper(freq="W-SAT"), "Sex"])["COVID-19 Deaths"]
    .sum()
    .unstack()
    .loc[date_range, "Male"]
)
female_official_us = (
    cdc.groupby([pd.Grouper(freq="W-SAT"), "Sex"])["COVID-19 Deaths"]
    .sum()
    .unstack()
    .loc[date_range, "Female"]
)

fig, ax = plt.subplots(figsize=(10, 5))
(male_official_us / female_official_us).plot(label="Male/Female deaths (CDC)")
(male_grouped_counts / female_grouped_counts).plot(
    label="Male/Female death tweets count (classifier)"
)

ax.set_ylabel("Male/Female ratio")
ax.legend()

# Convert plot to CMYK image
fig.canvas.draw()

img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

img = Image.fromarray(img_arr, mode="RGB")
img = img.convert("CMYK")
img = trim(img)
img.save(gender_series_plot_dir + "US_gender_ratio_series.pdf", dpi=(300, 300))

# pearson correlation
corr = pearsonr(
    (np.log(male_grouped_counts.values / female_grouped_counts.values)),
    (np.log(male_official_us.values / female_official_us.values)),
)

print("(Log) correlation: ")
print(
    f"({country}) tweets-official gender ratio correlation: {corr.statistic} (p-value={corr.pvalue})"
)

# for the UK
country = "United Kingdom"
freq = "W-FRI"
shift_period = 0
date_range = pd.date_range(start="2020-04-01", end="2021-03-31", freq=freq)

male_counts = df_with_gender[
    (df_with_gender.regex_gender == "male") & (df_with_gender.country_name == country)
]
female_counts = df_with_gender[
    (df_with_gender.regex_gender == "female") & (df_with_gender.country_name == country)
]

male_grouped_counts = male_counts.resample(freq).tweet_id.count()
female_grouped_counts = female_counts.resample(freq).tweet_id.count()

male_grouped_counts = male_grouped_counts.reindex(date_range, fill_value=0)
female_grouped_counts = female_grouped_counts.reindex(date_range, fill_value=0)

# Fixing date lag
male_grouped_counts = male_grouped_counts.shift(periods=shift_period, freq=freq)
female_grouped_counts = female_grouped_counts.shift(periods=shift_period, freq=freq)

# Get ONS data
df_male = pd.concat(
    [
        pd.read_csv(f"./data/official/ons_demographic/uk_male_deaths_{d}.csv")
        for d in ["2020", "2021"]
    ],
    axis=1,
).drop("Age group", axis=1)
df_male = df_male.apply(lambda x: x.apply(lambda y: int(str(y).replace(",", "")))).sum(
    axis=0
)
df_male.index = pd.to_datetime(df_male.index)

df_female = pd.concat(
    [
        pd.read_csv(f"./data/official/ons_demographic/uk_female_deaths_{d}.csv")
        for d in ["2020", "2021"]
    ],
    axis=1,
).drop("Age group", axis=1)
df_female = df_female.apply(
    lambda x: x.apply(lambda y: int(str(y).replace(",", "")))
).sum(axis=0)
df_female.index = pd.to_datetime(df_female.index)


male_official_uk = df_male.loc[male_grouped_counts.index]
female_official_uk = df_female.loc[female_grouped_counts.index]

# pearson correlation
corr = pearsonr(
    np.log(
        np.nan_to_num(
            male_grouped_counts.values / female_grouped_counts.values,
            nan=1e-8,
            posinf=1e-8,
        )
    ),
    np.log(male_official_uk.values / female_official_uk.values),
)

print("(Log) correlation: ")
print(
    f"({country}) tweets-official ratio correlation: {corr.statistic} (p-value={corr.pvalue})"
)
