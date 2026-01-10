from src.liability_generator.model_liability_management import (
    gan_train_and_save_model,
)
from src.liability_generator.model_liability import CTGAN
from pathlib import Path
from data.training_data import TRAINING_DATA_FOLDER_PATH
import pandas as pd


DATA_PATH = Path() / TRAINING_DATA_FOLDER_PATH / "payroll_LA_clean.csv"


DESTINATION_PATH_FOLDER = Path(__file__).resolve().parent

training_data = pd.read_csv(DATA_PATH)


def train_libaility_model(epochs: int, training_data: str) -> None:

    data = training_data.drop(["Unnamed: 0"], axis=1)
    data = data[data.index < 2001]
    model_name = "LA_model"
    categorical_columns = [
        "EMPLOYMENT_TYPE",
        "MOU_TITLE",
        "GENDER",
        "ETHNICITY",
    ]

    gan_train_and_save_model(
        model_name=model_name,
        categorical_columns=categorical_columns,
        model=CTGAN(),
        data=data,
        model_path_folder=DESTINATION_PATH_FOLDER,
        epochs=epochs,
    )


train_libaility_model(epochs=10, training_data=training_data)
