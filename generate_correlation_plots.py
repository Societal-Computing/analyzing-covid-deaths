import os
import random
import pycountry
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr, pearsonr

from tqdm import tqdm


from helpers.data import (
    get_official_deaths_CDC,
    get_official_deaths_UK,
    get_official_deaths_italy,
    get_official_deaths_australia,
    get_official_deaths_jhu,
    get_official_confirmed_jhu,
    get_official_deaths_google,
    get_official_deaths_who,
    get_excess_deaths_econ,
    load_covid_death_tweets,
    load_megacov_counts,
    load_tbcov_counts,
)

sns.set(font_scale=1.1)
sns.set_style("whitegrid", {"axes.grid": False})

# Seed for reproducing the experiment
random_seed = 42

random.seed(random_seed)
np.random.seed(random_seed)

CORRELATIONS_PLOT_DIR = "./out/plots/correlations/"
COVID_DEATH_TWEETS = "./out/data/regex_filtered_english_with_death_labels.csv"


COUNTRIES = ["Australia", "Canada", "India", "Italy", "United Kingdom", "United States"]
START_DATE = "2020-03-01"
END_DATE = "2021-03-31"

# the frequencies and shift periods to compute correlations for
frequencies = ["D", "W", "M"]
# shift_periods = {
#     "D": [-21, -14, -7, 0, 7, 14, 21],
#     "W": [-3, -2, -1, 0, 1, 2, 3],
#     "M": [-1, 0, 1],
# }
shift_periods = {
    "D": [0],
    "W": [0],
    "M": [0],
}


# load official data
# Demographic data, TBCOV and MegaCOV will be loaded for each country separately
global_covid_deaths_who = get_official_deaths_who()
global_covid_deaths_who["Date_reported"] = pd.to_datetime(
    global_covid_deaths_who.Date_reported
)

global_covid_deaths_google = get_official_deaths_google()
global_covid_deaths_google["date"] = pd.to_datetime(global_covid_deaths_google.date)

global_covid_deaths_jhu = get_official_deaths_jhu()
global_covid_confirmed_jhu = get_official_confirmed_jhu()

global_excess_deaths_econ = get_excess_deaths_econ()

us_deaths_cdc_weekly = get_official_deaths_CDC(weekly=True)
us_deaths_cdc_monthly = get_official_deaths_CDC(weekly=False)

uk_deaths_daily = get_official_deaths_UK()

# load covid death tweets
comb_df = load_covid_death_tweets(COVID_DEATH_TWEETS)


df_death_mentioned_with_date = comb_df[
    comb_df.does_the_tweet_refer_to_the_covidrelated_death_of_one_or_more_individuals_personally_known_to_the_tweets_author
    == "yes_label"
]


def plot_two_series(s1, s2, s3=None, l1="", l2="", l3="", country=""):
    """
    Plots two series on the same plot
    """

    country_plot_dir = CORRELATIONS_PLOT_DIR + f"{'_'.join(country.lower().split())}/"

    if not os.path.isdir(country_plot_dir):
        os.mkdir(country_plot_dir)

    title = f"{country}: {l1} vs {l2}"

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax2 = ax.twinx()
    ax3 = ax.twinx()

    p1 = ax.plot(s1.index, s1.values, label=l1)
    p2 = ax2.plot(s3.index, s3.values, "k:", label=l3)
    if s3 is not None:
        p3 = ax3.plot(s2.index, s2.values, "r", label=l2)
        ax3.set_ylabel(l2)
        # ax3.legend(loc="upper left")

    # ax.set_title(title)

    ax.set_ylabel(l1)
    ax2.set_ylabel(l3)

    ax.spines["right"].set_color("#1f77b4")
    ax.yaxis.label.set_color("#1f77b4")
    ax.tick_params(axis="y", colors="#1f77b4")

    if s3 is not None:
        ax.legend(handles=p1 + p2 + p3, loc="upper center")
        ax3.spines["right"].set_position(("outward", 60))
        # ax3.spines["right"].set_color("red")
        ax3.yaxis.label.set_color("red")
        ax3.tick_params(axis="y", colors="red")
    else:
        ax.legend(handles=p1 + p2, loc="upper center")

    # Convert plot to CMYK image
    fig.canvas.draw()

    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    img = Image.fromarray(img_arr, mode="RGB")
    img = img.convert("CMYK")
    img.save(
        CORRELATIONS_PLOT_DIR + f"{'_'.join(country.lower().split())}/{title}.pdf",
        dpi=(300, 300),
    )

    plt.close()
    return


masks = {country: dict({f: list() for f in frequencies}) for country in COUNTRIES}

data_temp = {
    "Death tweets (regex)": None,
    "Death tweets (clf)": None,
    "TBCOV": None,
    "MegaGeoCovExtended": None,
    "CSSE JHU (deaths)": None,
    "WHO (deaths)": None,
    "Covid-19 Open data (deaths)": None,
    "OWID (excess deaths)": None,
    "CSSE JHU (confirmed cases)": None,
}

data = {}


for country in tqdm(COUNTRIES):
    correlations = {}
    correlations_ranked = {}

    for freq in frequencies:
        # date range
        date_range = pd.date_range(start=START_DATE, end=END_DATE, freq=freq)

        for shift_period in shift_periods[freq]:
            # add data for given freq and shift
            data_key = f"data-freq={freq}-shift={shift_period}{freq}"
            data[data_key] = data_temp.copy()

            if country == "United States" and freq == "W":
                # fixing week definition for CDC, US
                freq = "W-SAT"
                date_range = pd.date_range(start=START_DATE, end=END_DATE, freq=freq)

            grouper = df_death_mentioned_with_date[
                df_death_mentioned_with_date.country_name == country
            ].resample(freq)
            grouped_counts = grouper.tweet_id.count()

            grouped_counts = grouped_counts.reindex(date_range, fill_value=0)

            # Get the shifted date range
            shifted_date_range = grouped_counts.index.shift(
                periods=shift_period, freq=freq
            )

            # simply get the counts of the death mentioned tweets i.e. regex filtered tweets
            regex_counts = (
                comb_df[comb_df.country_name == country].resample(freq).tweet_id.count()
            )
            regex_counts = regex_counts.reindex(date_range, fill_value=0)
            regex_counts = regex_counts.shift(periods=shift_period, freq=freq)

            # get hashtag counts for given country
            hashtag_counts = load_tbcov_counts(country, freq=freq, shift=shift_period)
            hashtag_counts["date_time"] = pd.to_datetime(hashtag_counts.date_time)
            hashtag_counts = hashtag_counts.set_index("date_time")
            hashtag_counts = hashtag_counts.reindex(shifted_date_range, fill_value=0)

            # get hashtag counts from MegaCovExtended for given country
            megacov_hashtag_counts = load_megacov_counts(
                country, freq=freq, shift=shift_period
            )
            megacov_hashtag_counts["created_at"] = pd.to_datetime(
                megacov_hashtag_counts.created_at
            )
            megacov_hashtag_counts = megacov_hashtag_counts.set_index("created_at")
            megacov_hashtag_counts = megacov_hashtag_counts.reindex(
                shifted_date_range, fill_value=0
            )

            # get death counts from WHO's data for given country
            #
            # make country correction
            corrected_country = country
            if corrected_country == "United States":
                corrected_country = "United States of America"
            elif corrected_country == "United Kingdom":
                corrected_country = "The United Kingdom"

            who_data = global_covid_deaths_who[
                global_covid_deaths_who.Country == corrected_country
            ]
            who_data = who_data.set_index("Date_reported")
            who_data = who_data.resample(freq).New_deaths.sum()
            who_data = who_data.reindex(shifted_date_range, fill_value=0)

            # get death counts from JHU's data for given country
            #
            # make country correction
            corrected_country = country
            if corrected_country == "United States":
                corrected_country = "US"

            jhu_data = global_covid_deaths_jhu.loc[:, corrected_country]
            jhu_data.index = pd.to_datetime(jhu_data.index)

            jhu_data = jhu_data.resample(freq).sum()
            jhu_data = jhu_data.reindex(shifted_date_range, fill_value=0)

            # get confirmed cases from JHU data
            #
            jhu_confirmed_data = global_covid_confirmed_jhu.loc[:, corrected_country]
            jhu_confirmed_data.index = pd.to_datetime(jhu_confirmed_data.index)
            jhu_confirmed_data = jhu_confirmed_data.resample(freq).sum()
            jhu_confirmed_data = jhu_confirmed_data.reindex(
                shifted_date_range, fill_value=0
            )

            # get death counts from google aggregated dataset
            #
            # make country correction
            corrected_country = pycountry.countries.search_fuzzy(country)[0].alpha_2

            google_data = global_covid_deaths_google[
                global_covid_deaths_google.location_key == corrected_country
            ]
            google_data = google_data.set_index("date")
            google_data = google_data.resample(freq).new_deceased.sum()
            google_data = google_data.reindex(shifted_date_range, fill_value=0)

            # get excess deaths data frmo OWID
            #
            excess_mortality_data = global_excess_deaths_econ[
                global_excess_deaths_econ.location == country
            ].set_index("date")
            excess_mortality_data.index = pd.to_datetime(excess_mortality_data.index)
            excess_mortality_data = excess_mortality_data.resample(
                freq
            ).excess_proj_all_ages.sum()
            excess_mortality_data = excess_mortality_data.reindex(
                shifted_date_range, fill_value=0
            )

            if country == "United States" and freq == "M":
                # This is for the CDC data
                us_deaths_cdc = get_official_deaths_CDC(weekly=False)

                us_deaths_cdc = us_deaths_cdc["COVID-19 Deaths"]

                us_deaths_cdc = us_deaths_cdc.reindex(shifted_date_range, fill_value=0)

                us_deaths_cdc.index.freq = "M"

                data[data_key]["CDC (deaths)"] = us_deaths_cdc.values

            elif country == "United States" and freq == "W-SAT":
                # This is for the CDC data
                us_deaths_cdc = get_official_deaths_CDC(weekly=True)

                us_deaths_cdc = us_deaths_cdc["COVID-19 Deaths"]

                us_deaths_cdc.index.freq = "W-SAT"

                us_deaths_cdc = us_deaths_cdc.reindex(shifted_date_range, fill_value=0)
                assert (
                    us_deaths_cdc.index.freq == "W-SAT"
                ), "Frequency mismatch (us_deaths_cdc)"

                data[data_key]["CDC (deaths)"] = us_deaths_cdc.values

                if freq == "W-SAT":
                    plot_two_series(
                        s1=grouped_counts,
                        s2=us_deaths_cdc,
                        s3=hashtag_counts,
                        country=country,
                        l1="Death tweets (clf)",
                        l2="CDC (deaths)",
                        l3="TBCOV",
                    )

            elif country == "United Kingdom":
                uk_df = uk_deaths_daily.resample(
                    freq
                ).newDailyNsoDeathsByDeathDate.sum()

                uk_df = uk_df.reindex(shifted_date_range, fill_value=0)

                data[data_key]["UKHSA (deaths)"] = uk_df.values

                if freq == "W":
                    plot_two_series(
                        s1=grouped_counts,
                        s2=uk_df,
                        s3=hashtag_counts,
                        country=country,
                        l1="Death tweets (clf)",
                        l2="UKHSA (deaths)",
                        l3="TBCOV",
                    )

            elif country == "Italy":
                italy_df = get_official_deaths_italy()

                italy_df = italy_df.resample(freq).deceduti.sum()

                italy_df = italy_df.reindex(shifted_date_range, fill_value=0)

                data[data_key]["CPD (deaths)"] = italy_df.values

            elif country == "Australia" and freq == "M":
                aus_df = get_official_deaths_australia()

                aus_df = aus_df.reindex(shifted_date_range, fill_value=0)

                data[data_key]["ABS (deaths)"] = aus_df.deaths.values

            # Add them to data for easy access
            data[data_key]["Death tweets (clf)"] = grouped_counts.values

            # get regex based counts
            data[data_key]["Death tweets (regex)"] = regex_counts.values

            # get hashtag counts from TBCOV
            data[data_key]["TBCOV"] = hashtag_counts.values.flatten()

            # get hashtag counts from megacov
            data[data_key][
                "MegaGeoCovExtended"
            ] = megacov_hashtag_counts.values.flatten()

            data[data_key]["CSSE JHU (deaths)"] = jhu_data
            # sum values from Provinces
            if isinstance(data[data_key]["CSSE JHU (deaths)"], pd.DataFrame):
                data[data_key]["CSSE JHU (deaths)"] = data[data_key][
                    "CSSE JHU (deaths)"
                ].sum(axis=1)
            data[data_key]["CSSE JHU (deaths)"] = data[data_key][
                "CSSE JHU (deaths)"
            ].values

            data[data_key]["CSSE JHU (confirmed cases)"] = jhu_confirmed_data
            # sum values from Provinces
            if isinstance(data[data_key]["CSSE JHU (confirmed cases)"], pd.DataFrame):
                data[data_key]["CSSE JHU (confirmed cases)"] = data[data_key][
                    "CSSE JHU (confirmed cases)"
                ].sum(axis=1)
            data[data_key]["CSSE JHU (confirmed cases)"] = data[data_key][
                "CSSE JHU (confirmed cases)"
            ].values

            data[data_key]["WHO (deaths)"] = who_data.values

            data[data_key]["Covid-19 Open data (deaths)"] = google_data.values

            data[data_key]["OWID (excess deaths)"] = excess_mortality_data.values

            labels = list(data[data_key].keys())

            # temporary df for storing correlation values
            corr_df = pd.DataFrame(columns=labels, index=labels)
            corr_ranked_df = pd.DataFrame(columns=labels, index=labels)

            # temporary df for storing p-values
            pvalues_df = pd.DataFrame(columns=labels, index=labels)
            pvalues_ranked_df = pd.DataFrame(columns=labels, index=labels)

            for l1 in labels:
                for l2 in labels:
                    corr = pearsonr(data[data_key][l1], data[data_key][l2])
                    corr_r = spearmanr(data[data_key][l1], data[data_key][l2])
                    corr_df.loc[l1, l2] = np.round(corr.statistic, 3)
                    corr_ranked_df.loc[l1, l2] = np.round(corr_r.statistic, 3)

            if freq == "W-SAT":
                # change frequency back to "W" instead of "W-SAT" for the US
                # to reuse the following parts of the code
                freq = "W"

            correlations[f"corr-freq={freq}-shift={shift_period}{freq}"] = corr_df
            correlations_ranked[
                f"corr-freq={freq}-shift={shift_period}{freq}"
            ] = corr_ranked_df

    # plot and save the correlation plots
    country_plot_dir = CORRELATIONS_PLOT_DIR + f"{'_'.join(country.lower().split())}/"

    if not os.path.isdir(country_plot_dir):
        os.mkdir(country_plot_dir)

    for freq in frequencies:
        freq_filter_phrase = f"corr-freq={freq}"

        ncols = len(shift_periods[freq])

        # plot linear correlation
        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(ncols * 10, 10),
            constrained_layout=True,
            sharey=False,
        )

        keys = [k for k in correlations.keys() if freq_filter_phrase in k]

        for j, key in enumerate(keys):
            mask = np.ones_like(correlations[key], dtype=bool)
            mask[np.tril_indices_from(mask, k=-1)] = False

            masks[country][freq].append(mask)

            annotations = np.char.add(correlations[key].values.astype(str), "")

            try:
                ax = axes[j]
            except:
                ax = axes

            g = sns.heatmap(
                correlations[key].astype(np.float32),
                annot=annotations,
                fmt="",
                annot_kws={"rotation": 45, "fontsize": 12},
                ax=ax,
                mask=mask,
            )
            ax.tick_params(axis="x", rotation=90)
            # ax.set_title(key)

        plt.suptitle(f"{country}")

        # Convert plot to CMYK image
        fig.canvas.draw()

        img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(img_arr, mode="RGB")
        img = img.convert("CMYK")
        img.save(
            country_plot_dir + f"correlation_linear_{country}-{freq}.pdf",
            dpi=(300, 300),
        )

        # fig.savefig(country_plot_dir + f"correlation_linear_{country}-{freq}.png")

        # plot ranked correlation
        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(ncols * 10, 10),
            constrained_layout=True,
            sharey=False,
        )

        try:
            ax = axes[j]
        except:
            ax = axes

        keys = [k for k in correlations_ranked.keys() if freq_filter_phrase in k]

        for j, key in enumerate(keys):
            mask = np.ones_like(correlations_ranked[key], dtype=bool)
            mask[np.tril_indices_from(mask, k=-1)] = False

            masks[country][freq].append(mask)

            annotations = np.char.add(correlations_ranked[key].values.astype(str), "")

            g = sns.heatmap(
                correlations_ranked[key].astype(np.float32),
                annot=annotations,
                fmt="",
                annot_kws={"rotation": 45, "fontsize": 14},
                ax=ax,
                mask=mask,
            )
            ax.tick_params(axis="x", rotation=90)
            # ax.set_title(key)

        plt.suptitle(f"{country}")

        # Convert plot to CMYK image
        fig.canvas.draw()

        img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(img_arr, mode="RGB")
        img = img.convert("CMYK")
        img.save(
            country_plot_dir + f"correlation_ranked_{country}-{freq}.pdf",
            dpi=(300, 300),
        )

        plt.close(fig)
