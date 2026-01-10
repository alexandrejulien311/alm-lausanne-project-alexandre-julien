from src.asset_generator.model_asset import ConditionalOneStepLSTM
from src.asset_generator.model_asset_management import (
    lstm_see_results,
)
from pathlib import Path
from data.training_data import TRAINING_DATA_FOLDER_PATH
from src.trained_model import TRAINED_MODEL_PATH

import pandas as pd
from src.asset_generator.brownian_motion import browninan_predictions
from src.liability_generator.model_liability_management import (
    gan_see_results,
)
from src.asset_generator.graph_asset_generation import save_graphs
from loguru import logger

DESTINATION_PATH_FOLDER = Path(__file__).resolve().parent

TRAINING_DATA_PATH_ASSET = (
    Path() / TRAINING_DATA_FOLDER_PATH / "merged_data_extended.csv"
)

TRAINING_DATA_PATH_LIABILITY = (
    Path() / TRAINING_DATA_FOLDER_PATH / "payroll_LA_clean.csv"
)


def main():

    # Asset generation - LSTM
    training_data = pd.read_csv(TRAINING_DATA_PATH_ASSET)
    model_name = "asset"
    model = ConditionalOneStepLSTM(asset_to_predict=4)

    # see Amundi paper : https://research-center.amundi.com/files/nuxeo/dl/eb84a66b-ebac-4901-8d70-70f3ed02a0d4?inline=&utm
    # for the choice of the date
    training_data = training_data.set_index("Date")
    training_data = training_data[training_data.index >= "2001-01-01"]

    gold_prediction = browninan_predictions(
        start_date="2024-01-01",
        number_of_values=50,
        new_day_in_month_list=[1, 10, 20],
        first_price=2034.04,
        asset_name="gold_price",
        mu=-0.08,
    )
    output_path = Path() / DESTINATION_PATH_FOLDER / "synthetic_asset.csv"
    lstm_see_results(
        model_path_folder=TRAINED_MODEL_PATH,
        model_name=model_name,
        model=model,
        input_data=gold_prediction,
        training_data=training_data,
        output_path=output_path,
    )

    data_generated = pd.read_csv(output_path)
    graph_path = Path() / DESTINATION_PATH_FOLDER / "asset_graph.png"
    save_graphs(
        input_data=training_data[training_data.index >= "2010-01-01"],
        input_data_generated=data_generated,
        path=graph_path,
    )

    # ----------------

    # Liabilities generation

    training_data = pd.read_csv(TRAINING_DATA_PATH_LIABILITY)

    training_data = training_data.drop(["Unnamed: 0"], axis=1)
    training_data = training_data[training_data.index < 2001]
    model_name = "LA_model"
    categorical_columns = [
        "EMPLOYMENT_TYPE",
        "MOU_TITLE",
        "GENDER",
        "ETHNICITY",
    ]

    gan_see_results(
        model_path_folder=TRAINED_MODEL_PATH,
        model_name=model_name,
        amount_number=20,
        categorical_columns=categorical_columns,
        data=training_data,
        destination_path_folder=DESTINATION_PATH_FOLDER,
    )

if __name__ == '__main__':
    logger.info('main.py lauched')
    main()
