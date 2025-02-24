"""
Module for loading and saving dataset.
"""

import pandas as pd
from typing import Any

import logging

from .config import CONFIG

logger = logging.getLogger(__name__)


class DataLoader:
    """Class to load and save the knowledge base data."""

    def __init__(self, csv_path: str = CONFIG.KNOWLEDGE_BASE_PATH) -> None:
        """
        Args:
            csv_path: Path to the CSV file containing the knowledge base.
        """
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        """Loads the knowledge base dataset from a CSV file.

        Returns:
            A pandas DataFrame containing the dataset.

        Raises:
            FileNotFoundError: If the CSV file is not found.
        """
        try:
            df = pd.read_csv(self.csv_path)
            logger.info("Knowledge base loaded successfully from %s.", self.csv_path)
            return df
        except FileNotFoundError as e:
            logger.error("CSV file not found at %s", self.csv_path)
            raise e

    def save(self, df: pd.DataFrame, path: str) -> None:
        """Saves a DataFrame to a CSV file.

        Args:
            df: The pandas DataFrame to save.
            path: The file path where to save the CSV.
        """
        df.to_csv(path, index=False)
        logger.info("DataFrame saved to %s.", path)
