import torch
import pandas as pd
import numpy as np

# price of the last 6 month considering 3 values by month : 3*6
WINDOW_LENGTH = 18


class ConditionalSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, window: int = WINDOW_LENGTH):
        self.data = data
        self.window = window

    def __len__(self):
        return self.data.shape[0] - self.window - 1

    def __getitem__(self, idx):
        """
        Considering 5 assets : see comments about shape.
        """
        t = idx + self.window

        past_prices = self.data[t - self.window : t + 1]  # (window+1, 5)
        p1_next = self.data[t + 1, 0:1]  # (1,)
        y = self.data[t + 1, 1:]  # (4,)

        return past_prices, p1_next, y


def data_refactoring(
    input_data: pd.DataFrame, mu: pd.DataFrame, sigma: pd.DataFrame
) -> torch.Tensor:
    """
    Transform data to load them into the model (avoid sclaes problems between assets)
    """
    log_prices = np.log(input_data)
    log_returns = log_prices.diff().dropna()

    df = (log_returns - mu) / sigma

    results_tensor = torch.from_numpy(df.values).float()
    return results_tensor


def inverse_data_refactoring(
    input_tensor: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    initial_price: torch.Tensor,
    date_serie: pd.Series,
    column_names: list[str],
) -> pd.DataFrame:

    log_returns = input_tensor * sigma + mu

    log_initial_price = torch.log(initial_price)
    log_prices = torch.vstack(
        [log_initial_price, log_initial_price + torch.cumsum(log_returns, dim=0)]
    )

    prices_tensor = torch.exp(log_prices)
    df = pd.DataFrame(
        data=prices_tensor.detach().numpy(), index=date_serie, columns=column_names
    )

    return df
