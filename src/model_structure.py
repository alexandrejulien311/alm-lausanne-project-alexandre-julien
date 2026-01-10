from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):

    @abstractmethod
    def training_loop(self, epochs: int) -> None:
        pass

    @abstractmethod
    def save_model(self, folder_path: str) -> None:
        """
        Create a file to be able to reuse a model already trained.
        """
        pass

    @abstractmethod
    def generate(self, input) -> pd.DataFrame:
        pass
