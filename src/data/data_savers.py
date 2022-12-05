import pandas as pd
from pathlib import Path


class DataSaver:
    def save(self, content: list[pd.DataFrame],
             save_dir: Path) -> None:
        """

        params:
            content: list[pd.DataFrame] - list of dataframes with predicts
        return: None
        """

        for index, df in enumerate(content):
            df.to_csv(save_dir / f"test_{index + 1}.csv", index=False)
