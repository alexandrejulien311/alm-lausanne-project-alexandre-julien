import pandas as pd
from pathlib import Path

pd.options.mode.copy_on_write = True


def data_merger(path_list: list[Path]) -> pd.DataFrame:
    """
    Merge panda Series indexed with dates and indexed it with integers. Keep one column for date.
    """
    df_list = [pd.read_csv(path).set_index("Date") for path in path_list]
    merged_data = pd.concat(df_list, axis=1)
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
    return merged_data.reset_index()


def calibrate_date(data: pd.DataFrame, new_day_in_month_list: list[int]) -> pd.Series:
    """
    Return new dates in panda Series form so that we have the 1st, 10th and 20th for each month (only the 1st previously).
    """
    new_dates = []
    previous_date = pd.date_range(
        start=data["Date"].min(), end=data["Date"].max(), freq="MS"
    )

    for date in previous_date:
        new_dates.append(date)
        for increment_day in new_day_in_month_list:
            new_dates.append(date + pd.Timedelta(days=increment_day))
    new_dates = pd.to_datetime(sorted(new_dates))

    new_dates = new_dates[
        (new_dates >= data["Date"].min()) & (new_dates <= data["Date"].max())
    ]
    return new_dates


def shape_build(merged_data: pd.DataFrame, new_dates: pd.Series) -> pd.DataFrame:
    """
    New indexes, new values filled with None.
    """
    merged_data = merged_data.copy()
    merged_data["Date"] = pd.to_datetime(merged_data["Date"])
    new_dates = pd.to_datetime(new_dates)

    df_indexed = merged_data.set_index("Date")

    df_reindexed = df_indexed.reindex(new_dates)

    data_incomplete_df = df_reindexed.reset_index().rename(columns={"index": "Date"})

    return data_incomplete_df


def fill_none_values(data_incomplete_df: pd.DataFrame, method="linear") -> pd.DataFrame:
    new_columns_list = [data_incomplete_df["Date"]]
    for column_name in data_incomplete_df.columns:
        if column_name != "Date":
            asset = data_incomplete_df[column_name]
            asset = asset.interpolate(method=method, limit_direction="forward", axis=0)
            for i, value in enumerate(asset.values):
                asset.iloc[i] = round(number=value, ndigits=2)
            new_columns_list.append(asset)

    data_extended = pd.concat(new_columns_list, axis=1)
    data_extended["Date"] = pd.to_datetime(data_extended["Date"])
    return data_extended


def merge_interpolate(
    path_list: list[Path], new_day_in_month_list: list[int]
) -> pd.DataFrame:
    merged_data = data_merger(path_list)

    new_dates = calibrate_date(merged_data, new_day_in_month_list)
    data_incomplete_df = shape_build(merged_data, new_dates)

    merged_data_extended = fill_none_values(data_incomplete_df)
    merged_data_extended.set_index("Date")
    return merged_data_extended
