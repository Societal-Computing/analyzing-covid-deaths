import pandas as pd

OFFICIAL_DATA_DIR = "./data/official/"
COMPUTED_DATA_DIR = "./data/computed/"
COVID_TWEETS_DIR = "./out/data/"


# Official data from the CDC
def get_official_deaths_CDC(weekly=True):
    if weekly:
        us_cdc = pd.read_csv(
            OFFICIAL_DATA_DIR + "Provisional_COVID-19_Deaths_by_Week__Sex__and_Age.csv"
        )
    else:
        us_cdc = pd.read_csv(
            OFFICIAL_DATA_DIR + "Provisional_COVID-19_Deaths_by_Sex_and_Age.csv"
        )

    us_cdc = us_cdc[us_cdc.State == "United States"].reset_index(drop=True)

    age_key = "All Ages"
    us_cdc = us_cdc[us_cdc["Age Group"] == age_key].reset_index(drop=True)

    gender_key = "All Sex" if weekly else "All Sexes"
    us_cdc = us_cdc[us_cdc["Sex"] == gender_key].reset_index(drop=True)

    if not weekly:
        # if we aren't looking at weekly data, get monthly deaths
        # note: there's also the yearly counts
        us_cdc = us_cdc[us_cdc.Group == "By Month"].reset_index(drop=True)

    key = "End Week" if weekly else "End Date"

    us_cdc[key] = pd.to_datetime(us_cdc[key])

    us_cdc.index = us_cdc[key]

    return us_cdc


# Official deaths for Italy (CPD)
def get_official_deaths_italy():
    italy_cpd = pd.read_csv(
        OFFICIAL_DATA_DIR + "dpc-covid19-ita-andamento-nazionale.csv"
    )

    italy_cpd.data = pd.to_datetime(italy_cpd.data)

    italy_cpd = italy_cpd.set_index("data")

    italy_cpd_fr = italy_cpd.loc[italy_cpd.index[0]]
    italy_cpd.deceduti = italy_cpd.deceduti.diff()
    italy_cpd.loc[italy_cpd.index[0]] = italy_cpd_fr

    return italy_cpd


# Official deaths for Australia
def get_official_deaths_australia():
    australia_df = pd.read_csv(OFFICIAL_DATA_DIR + "australia_monthly_2020_2021.csv")

    australia_df.date = pd.to_datetime(australia_df.date)

    australia_df.index = australia_df.date

    return australia_df


# Official data from the NHS
def get_official_deaths_UK():
    uk_df = pd.read_csv(OFFICIAL_DATA_DIR + "UK_official_daily_deaths.csv")

    uk_df.date = pd.to_datetime(uk_df.date)

    uk_df.index = uk_df.date

    return uk_df


# Official data from JHU
def get_official_deaths_jhu():
    # Global data of Covid deaths from JHU
    # link: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
    global_covid_death_data = (
        OFFICIAL_DATA_DIR + "time_series_covid19_deaths_global.csv"
    )
    global_covid_death_data = pd.read_csv(global_covid_death_data)
    global_data_countries = global_covid_death_data["Country/Region"]

    global_covid_death_data = global_covid_death_data.iloc[:, 4:]
    global_covid_death_data = global_covid_death_data.T
    global_covid_death_data.columns = global_data_countries

    # keep track of first row, to fill in after restoring values from cumulative sum
    global_first_row = global_covid_death_data.iloc[0, :]

    global_covid_death_data_jhu = global_covid_death_data.diff(axis=0)
    global_covid_death_data_jhu.iloc[0, :] = global_first_row
    return global_covid_death_data_jhu


def get_official_confirmed_jhu():
    # Global data of confirmed covid cases from JHU
    # link: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
    global_covid_confirmed_data = (
        OFFICIAL_DATA_DIR + "time_series_covid19_confirmed_global.csv"
    )
    global_covid_confirmed_data = pd.read_csv(global_covid_confirmed_data)
    global_confirmed_data_countries = global_covid_confirmed_data["Country/Region"]

    global_covid_confirmed_data = global_covid_confirmed_data.iloc[:, 4:]
    global_covid_confirmed_data = global_covid_confirmed_data.T
    global_covid_confirmed_data.columns = global_confirmed_data_countries

    # keep track of first row, to fill in after restoring values from cumulative sum
    global_first_row = global_covid_confirmed_data.iloc[0, :]

    global_covid_confirmed_data_jhu = global_covid_confirmed_data.diff(axis=0)
    global_covid_confirmed_data_jhu.iloc[0, :] = global_first_row
    return global_covid_confirmed_data_jhu


# Official data from WHO
def get_official_deaths_who():
    # Global data of deaths from WHO
    global_covid_death_data_who = pd.read_csv(
        OFFICIAL_DATA_DIR + "WHO-COVID-19-global-data.csv"
    )
    return global_covid_death_data_who


# Official data aggregated by Google
def get_official_deaths_google():
    # Global deaths data aggregated by Google
    global_covid_death_data_google = pd.read_csv(
        OFFICIAL_DATA_DIR + "google_covid_deaths_various_sources.csv"
    )
    return global_covid_death_data_google


def get_excess_deaths_econ():
    # Excess mortality data Economist
    # link: https://github.com/owid/covid-19-data/blob/master/public/data/excess_mortality/
    global_covid_excess_death_data = OFFICIAL_DATA_DIR + "excess_mortality.csv"
    global_covid_excess_death_data_econ = pd.read_csv(global_covid_excess_death_data)
    return global_covid_excess_death_data_econ


# Get precomputed hashtag counts
def load_tbcov_counts(
    country, freq="W", shift=0, counts_dir=COMPUTED_DATA_DIR + "tbcov_counts"
):
    country_counts_dir = f"{counts_dir}/{'_'.join(country.lower().split())}"
    file_name = (
        country_counts_dir
        + f"/{'_'.join(country.lower().split())}-freq={freq}-shift={shift}{freq}.csv"
    )
    df = pd.read_csv(file_name)
    return df


def load_megacov_counts(
    country, freq="W", shift=0, megacov_counts_dir=COMPUTED_DATA_DIR + "megacov_counts"
):
    country_counts_dir = f"{megacov_counts_dir}/{'_'.join(country.lower().split())}"
    file_name = (
        country_counts_dir
        + f"/{'_'.join(country.lower().split())}-freq={freq}-shift={shift}{freq}.csv"
    )
    df = pd.read_csv(file_name)
    return df


def load_covid_death_tweets(covid_file_dir):
    df = pd.read_csv(covid_file_dir)
    df.date = pd.to_datetime(df.date)
    df = df.set_index(df.date)
    return df
