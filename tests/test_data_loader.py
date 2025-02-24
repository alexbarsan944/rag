import os
import pandas as pd
import pytest

from src.data_loader import DataLoader


def test_load_valid_csv(tmp_path):
    data = "Text\nHello World"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(data)

    loader = DataLoader(csv_path=str(csv_file))
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert "Text" in df.columns


def test_load_invalid_csv():
    loader = DataLoader(csv_path="non_existent.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()
