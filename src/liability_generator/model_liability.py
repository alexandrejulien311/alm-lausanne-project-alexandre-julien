from ctgan import CTGAN
from pathlib import Path
import pandas as pd
from loguru import logger
from src.model_structure import BaseModel


class DataTableGenerator(BaseModel):
    def __init__(
        self,
        name: str,
        model: CTGAN,
        data: pd.DataFrame,
        categorical_columns: list[str],
    ):
        self.categorical_columns = categorical_columns
        self.data = data
        self.model = model
        self.name = name

    def training_loop(self, epochs: int) -> None:
        logger.info("Training starts.")
        self.model.fit(self.data, self.categorical_columns, epochs=epochs)

    def save_model(self, folder_path: str) -> None:
        file_name = self.name + "_ctgan_model.pkl"
        path = Path(folder_path) / file_name
        self.model.save(path)

    def generate(self, input: int) -> pd.DataFrame:
        return self.model.sample(input)
