from src.asset_generator.model_asset import ConditionalOneStepLSTM

from src.asset_generator.model_asset_management import (
    lstm_train_and_save_model,
)
from pathlib import Path
from data.training_data import TRAINING_DATA_FOLDER_PATH
import pandas as pd


TRAINING_DATA_PATH = Path() / TRAINING_DATA_FOLDER_PATH / "merged_data_extended.csv"
DESTINATION_PATH_FOLDER = Path(__file__).resolve().parent

training_data = pd.read_csv(TRAINING_DATA_PATH)


def train_model_adsset(epochs: int, training_data) -> None:
    training_data = pd.read_csv(TRAINING_DATA_PATH)
    model_name = "asset"
    model = ConditionalOneStepLSTM(asset_to_predict=4)

    # see Amundi paper : https://research-center.amundi.com/files/nuxeo/dl/eb84a66b-ebac-4901-8d70-70f3ed02a0d4?inline=&utm
    training_data = training_data.set_index("Date")
    training_data = training_data[training_data.index >= "2001-01-01"]

    lstm_train_and_save_model(
        model_name=model_name,
        model=model,
        training_data=training_data,
        model_path_folder=DESTINATION_PATH_FOLDER,
        epochs=epochs,
    )


train_model_adsset(epochs=50, training_data=training_data)
