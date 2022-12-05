import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self):
        self.data_dir: Path = Path('')
        self.train_data: pd.DataFrame = pd.DataFrame()
        self.test_data: pd.DataFrame = pd.DataFrame()
        self.players_data: pd.DataFrame = pd.DataFrame()

    def load_data(self, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and return train, test and players data from data_dir

        params:
            data_dir: Path - path to directory with raw data
        return:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] - train_data, players_data, test_data
        """
        self.data_dir = data_dir
        self._load_data_from_csv()

        return self.train_data, self.players_data, self.test_data

    def _load_data_from_csv(self) -> None:
        """
        Load each .csv file from raw data directory
        """
        self.train_data = pd.read_csv(self.data_dir / "train.csv")
        self.test_data = pd.read_csv(self.data_dir / "test.csv")
        self.players_data = pd.read_csv(self.data_dir / "players_feats.csv")
