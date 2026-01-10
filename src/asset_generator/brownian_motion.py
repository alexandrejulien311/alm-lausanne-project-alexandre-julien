import torch
import pandas as pd


def generate_prices(
    first_price=100.0, mu=0.05, sigma=0.2, output_price_number=10, device="cpu"
) -> torch.Tensor:
    dt = 1.0 / output_price_number
    eps = torch.randn(output_price_number, device=device)

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * torch.sqrt(
        torch.tensor(dt)
    ) * eps

    prices = first_price * torch.exp(torch.cumsum(log_returns, dim=0))
    return prices


def generate_date(
    start_date: str, number_of_values: int, new_day_in_month_list: list[int]
) -> pd.Series:
    start = pd.to_datetime(start_date)

    dates = []

    for _ in range(
        number_of_values // len(new_day_in_month_list) + len(new_day_in_month_list)
    ):
        year = start.year
        month = start.month

        for day in new_day_in_month_list:
            try:

                new_date = pd.Timestamp(year=year, month=month, day=day)
                dates.append(new_date)
            except ValueError:
                pass
        start += pd.offsets.MonthBegin(1)

    series = pd.Series(dates, name="Date")
    series = series[series > start_date]
    series = series[:number_of_values]
    series = series.reset_index()
    return series


def browninan_predictions(
    start_date,
    number_of_values: int,
    new_day_in_month_list: list[int],
    first_price: float,
    asset_name: str,
    mu=0.05,
    sigma=0.2,
    device="cpu",
) -> pd.DataFrame:
    date_serie = generate_date(start_date, number_of_values, new_day_in_month_list)
    value_asset = generate_prices(
        first_price=first_price,
        mu=mu,
        sigma=sigma,
        output_price_number=number_of_values,
        device=device,
    )

    df = pd.DataFrame(data=value_asset, columns=[asset_name], index=date_serie["Date"])
    return df
