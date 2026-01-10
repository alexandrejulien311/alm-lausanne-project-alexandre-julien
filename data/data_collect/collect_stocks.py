from pathlib import Path
import pandas as pd
from data.dataset_raw import DATASET_RAW_PATH
from data.dataset_clean import DATASET_CLEAN_PATH

# https://www.macrotrends.net/datasets/2324/sp-500-historical-chart-data

START_DATE = "1965-01-01"
END_DATE = "2024-01-01"

path_sp500_raw = Path() / DATASET_RAW_PATH / "chart_sp500.csv"

data_sp500_value = pd.read_csv(
    path_sp500_raw, sep=";", decimal=",", parse_dates=["Date"], dayfirst=False
)

data_sp500_value.set_index("Date", inplace=True)
data_sp500_value.index.name = "Date"

data_sp500_value.index = data_sp500_value.index + pd.offsets.MonthBegin(1)

data_sp500_value = data_sp500_value.truncate(before=START_DATE)
data_sp500_value = data_sp500_value.truncate(after=END_DATE)

data_sp500_value = data_sp500_value.rename(columns={"Value": "sp500_points"})
path_sp500 = Path() / DATASET_CLEAN_PATH / "sp500_values.csv"
data_sp500_value.to_csv(path_sp500)
