import torch
import torch.nn as nn
import pandas as pd
from loguru import logger
from pathlib import Path
from src.model_structure import BaseModel
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.asset_generator.data_loader_asset import (
    ConditionalSequenceDataset,
    data_refactoring,
    inverse_data_refactoring,
)

# price of the last 6 month considering 3 values by month : 3*6
WINDOW_LENGTH = 18


class ConditionalOneStepLSTM(nn.Module):
    def __init__(self, asset_to_predict=int, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1 + asset_to_predict, hidden_size=hidden, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, asset_to_predict),
        )

    def forward(self, past_prices: torch.Tensor, p1_next: torch.Tensor):
        """
        Considering 5 assets for examples:
        past_prices shape: [5]
        p1_next shape: [1]
        """
        _, (h_T, _) = self.lstm(past_prices)
        h_T = h_T[-1]  # (batch, hidden)

        x = torch.cat([h_T, p1_next], dim=-1)
        return self.head(x)


def train_step(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    loss_fn=nn.MSELoss(),
) -> None:

    model.train()
    epoch_loss = 0
    for past_prices, p1_next, y in loader:

        past_prices = past_prices.float()
        p1_next = p1_next.float()
        y = y.float()

        optimizer.zero_grad()
        y_pred = model(past_prices, p1_next)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    avg_loss = epoch_loss / len(loader)

    if epoch % 10 == 0:
        logger.info(f"Epoch {epoch}: loss={avg_loss:.6f}")


class AssetGenerator(BaseModel):
    def __init__(
        self,
        name: str,
        training_data: pd.DataFrame,
        model,
        folder_path_parameters: Path,
    ):
        self.training_data = (
            training_data.set_index("Date")
            if "Date" in training_data.columns
            else training_data
        )
        self.name = name
        self.model = model
        self.folder_path_parameters = folder_path_parameters

        log_prices = np.log(self.training_data)
        log_returns = log_prices.diff().dropna()

        self.mu = log_returns.mean()
        self.sigma = log_returns.std()

    def training_loop(self, epochs: int) -> tuple[float, float]:
        """
        Take the first column of asset as base to predict the rests of the asset's values.
        Thus, if you want to make a prediction, you have to set hypothetic values for the first asset.
        """
        logger.info("Training starts.")

        data_tensor = data_refactoring(
            input_data=self.training_data, mu=self.mu, sigma=self.sigma
        )
        dataset = ConditionalSequenceDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        for epoch in tqdm(range(epochs)):
            train_step(
                model=self.model, loader=loader, epoch=epoch, optimizer=optimizer
            )

    def save_model(self) -> None:
        file_name = self.name + "_lstm_model.pth"
        path = Path(self.folder_path_parameters) / file_name
        torch.save(self.model.state_dict(), path)

    def generate(self, input: pd.DataFrame) -> pd.DataFrame:
        """
        Considering 5 assets :
        - first one in input predicted for n times using brownian motion for example.
        - rest of asset (4 here) generated using the model.
        """

        last_line_in_training_data = self.training_data.tail(1)

        first_asset_predictions_bm = data_refactoring(
            input_data=input, mu=self.mu[0], sigma=self.sigma[0]
        )
        model_predictions = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        past_prices_df = self.training_data.tail(WINDOW_LENGTH)
        past_prices_all_assets = data_refactoring(
            past_prices_df, mu=self.mu, sigma=self.sigma
        )

        self.model.eval()
        for i in range(len(input) - 1):
            rest_asset_predictions = self.model(
                past_prices=past_prices_all_assets,
                p1_next=first_asset_predictions_bm[i],
            )
            one_step_predictions = torch.cat(
                (first_asset_predictions_bm[i], rest_asset_predictions), dim=0
            )
            past_prices_all_assets = torch.cat(
                (past_prices_all_assets[1:], one_step_predictions.unsqueeze(0)), dim=0
            )
            model_predictions = torch.cat(
                (model_predictions, rest_asset_predictions.unsqueeze(0)), dim=0
            )

        model_predictions = model_predictions[1:]
        total_predictions = torch.cat(
            (first_asset_predictions_bm, model_predictions), dim=1
        )

        clean_predictions = inverse_data_refactoring(
            input_tensor=total_predictions,
            initial_price=torch.tensor(last_line_in_training_data.values),
            mu=torch.tensor(self.mu.values),
            sigma=torch.tensor(self.sigma.values),
            date_serie=input.index,
            column_names=self.training_data.columns,
        )

        return clean_predictions.round(2)
