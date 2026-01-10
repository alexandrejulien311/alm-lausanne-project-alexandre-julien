from pathlib import Path
import pandas as pd
from data.dataset_raw import DATASET_RAW_PATH
from data.dataset_clean import DATASET_CLEAN_PATH

# https://www.macrotrends.net/2522/5-year-treasury-bond-rate-yield-chart

START_DATE = "1965-01-01"
END_DATE = "2024-01-01"


# 1 year bonds
path_bonds_1y = Path() / DATASET_RAW_PATH / "chart_american_bonds_1Y.csv"
data_bonds_1y = pd.read_csv(
    path_bonds_1y, sep=";", decimal=",", parse_dates=["Date"], dayfirst=False
)

data_bonds_1y.set_index("Date", inplace=True)
data_bonds_1y.index.name = "Date"

data_bonds_1y = data_bonds_1y.truncate(before=START_DATE)
data_bonds_1y = data_bonds_1y.truncate(after=END_DATE)

data_bonds_1y = data_bonds_1y.rename(columns={"Value": "rate_1Y"})
data_bonds_1y = data_bonds_1y.resample("MS").first()
file_path = Path() / DATASET_CLEAN_PATH / "data_bonds_1y.csv"
data_bonds_1y.to_csv(file_path)


# 5 year bonds
path_bonds_5y = Path() / DATASET_RAW_PATH / "chart_american_bonds_10Y.csv"
data_bonds_5y = pd.read_csv(
    path_bonds_5y, sep=";", decimal=",", parse_dates=["Date"], dayfirst=False
)

data_bonds_5y.set_index("Date", inplace=True)
data_bonds_5y.index.name = "Date"

data_bonds_5y = data_bonds_5y.truncate(before=START_DATE)
data_bonds_5y = data_bonds_5y.truncate(after=END_DATE)

data_bonds_5y = data_bonds_5y.rename(columns={"Value": "rate_5Y"})
data_bonds_5y = data_bonds_5y.resample("MS").first()
file_path = Path() / DATASET_CLEAN_PATH / "data_bonds_5y.csv"
data_bonds_5y.to_csv(file_path)


# 10 year bonds
path_bonds_10y = Path() / DATASET_RAW_PATH / "chart_american_bonds_5Y.csv"
data_bonds_10y = pd.read_csv(
    path_bonds_10y, sep=";", decimal=",", parse_dates=["Date"], dayfirst=False
)

data_bonds_10y.set_index("Date", inplace=True)
data_bonds_10y.index.name = "Date"

data_bonds_10y = data_bonds_10y.truncate(before=START_DATE)
data_bonds_10y = data_bonds_10y.truncate(after=END_DATE)

data_bonds_10y = data_bonds_10y.rename(columns={"Value": "rate_10Y"})
data_bonds_10y = data_bonds_10y.resample("MS").first()
file_path = Path() / DATASET_CLEAN_PATH / "data_bonds_10y.csv"
data_bonds_10y.to_csv(file_path)
