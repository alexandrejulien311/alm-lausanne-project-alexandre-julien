import pandas as pd
from pathlib import Path
from data.dataset_raw import DATASET_RAW_PATH
from data.dataset_clean import DATASET_CLEAN_PATH

# not working anymore ... : https://datahub.io/core/gold-prices/r/monthly.csv

START_DATE = "1965-01-01"
END_DATE = "2024-01-01"


path_gold_price_monthly = Path() / DATASET_RAW_PATH / "gold_price_monthly.csv"

data_gold_price = pd.read_csv(path_gold_price_monthly, parse_dates=["Date"])
data_gold_price.set_index("Date", inplace=True)


data_gold_price = data_gold_price.truncate(before=START_DATE)
data_gold_price = data_gold_price.truncate(after=END_DATE)

data_gold_price = data_gold_price.rename(columns={"Price": "gold_price"})
file_path = Path() / DATASET_CLEAN_PATH / "gold_prices.csv"
data_gold_price.to_csv(file_path)
