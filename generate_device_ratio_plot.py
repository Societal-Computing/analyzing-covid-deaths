import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

from helpers.data import load_covid_death_tweets


def trim(im):
    # helper for trimming the plots
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


PLOT_DIR = "./out/plots/"
COVID_DEATH_TWEETS = "./out/data/regex_filtered_english_with_death_labels.csv"

date_range = pd.to_datetime(
    [
        "2020-03-31",
        "2020-04-30",
        "2020-05-31",
        "2020-06-30",
        "2020-07-31",
        "2020-08-31",
        "2020-09-30",
        "2020-10-31",
        "2020-11-30",
        "2020-12-31",
        "2021-01-31",
        "2021-02-28",
        "2021-03-31",
    ]
)

df = load_covid_death_tweets(COVID_DEATH_TWEETS)

df = df[
    df.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
    == "yes_label"
].reset_index(drop=True)

df_with_devices = df[~df.matched_device.isna()].reset_index(drop=True)
df_with_devices.date = pd.to_datetime(df_with_devices.date)
df_with_devices.index = df_with_devices.date

# for the US
country = "United States"
data_us = (
    df_with_devices[df_with_devices.country_name == country]
    .groupby([pd.Grouper(freq="M"), "user_gender"])
    .matched_device.value_counts()
    .unstack()
    .unstack()
    .loc[date_range]
)

# for the UK
country = "United Kingdom"
data_uk = (
    df_with_devices[df_with_devices.country_name == country]
    .groupby([pd.Grouper(freq="M"), "user_gender"])
    .matched_device.value_counts()
    .unstack()
    .unstack()
    .loc[date_range]
)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

ax.set_ylabel("Android-to-Apple ratio")

(data_us["Android"].sum(axis=1) / data_us["Apple"].sum(axis=1))[date_range].plot(
    ax=ax, label="United States"
)
(data_uk["Android"].sum(axis=1) / data_uk["Apple"].sum(axis=1))[date_range].plot(
    ax=ax, label="United Kingdom"
)

ax.legend()

# Convert plot to CMYK image
fig.canvas.draw()

img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

img = Image.fromarray(img_arr, mode="RGB")
img = img.convert("CMYK")
img = trim(img)
img.save(
    PLOT_DIR + "device_ratio.pdf",
    dpi=(300, 300),
)
