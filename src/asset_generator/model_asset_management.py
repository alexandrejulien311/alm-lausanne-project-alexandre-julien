from pathlib import Path
import pandas as pd
from src.asset_generator.model_asset import AssetGenerator
import torch


def lstm_train_and_save_model(
    model_name: str,
    model,
    training_data: pd.DataFrame,
    model_path_folder: Path,
    epochs: int,
) -> None:

    asset_generator = AssetGenerator(
        name=model_name,
        training_data=training_data,
        model=model,
        folder_path_parameters=model_path_folder,
    )
    asset_generator.training_loop(epochs)
    asset_generator.save_model()


def lstm_see_results(
    model_path_folder: Path,
    model_name: str,
    model,
    input_data: pd.DataFrame,
    training_data: pd.DataFrame,
    output_path: str,
) -> None:
    file_name = model_name + "_lstm_model.pth"
    path = Path() / model_path_folder / file_name
    model.load_state_dict(torch.load(path))

    asset_generator = AssetGenerator(
        name=model_name,
        training_data=training_data,
        model=model,
        folder_path_parameters=model_path_folder,
    )
    synthetic_data_df = asset_generator.generate(input_data)
    synthetic_data_df.to_csv(output_path)
