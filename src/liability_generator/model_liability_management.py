from src.liability_generator.model_liability import (
    DataTableGenerator,
    CTGAN,
)
import pandas as pd
from pathlib import Path


def gan_train_and_save_model(
    model_name: str,
    categorical_columns: list[str],
    model: CTGAN,
    data: pd.DataFrame,
    model_path_folder: Path,
    epochs: int,
) -> None:

    data_table_generator = DataTableGenerator(
        name=model_name,
        categorical_columns=categorical_columns,
        data=data,
        model=model,
    )
    data_table_generator.training_loop(epochs)
    data_table_generator.save_model(model_path_folder)


def gan_see_results(
    model_path_folder: Path,
    model_name: str,
    amount_number,
    categorical_columns: list[str],
    data: pd.DataFrame,
    destination_path_folder: str,
) -> None:
    file_gan_name = model_name + "_ctgan_model.pkl"
    gan_path = Path() / model_path_folder / file_gan_name
    ctgan = CTGAN.load(gan_path)

    data_table_generator = DataTableGenerator(
        name=model_name,
        categorical_columns=categorical_columns,
        data=data,
        model=ctgan,
    )
    synthetic_data_df = data_table_generator.generate(amount_number)

    file_synthetic_data_name = model_name + "_synthetic_data.csv"
    synthetic_data_path = Path() / destination_path_folder / file_synthetic_data_name
    synthetic_data_df.to_csv(synthetic_data_path)
