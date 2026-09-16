from pathlib import Path

import pandas as pd
import pytest

from src.data_preprocessing import (
    clean_dataset,
    load_and_clean_dataset,
    load_raw_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"


def test_dataset_file_exists():
    assert DATASET_PATH.exists()
    assert DATASET_PATH.is_file()


def test_load_raw_dataset():
    dataframe = load_raw_dataset(DATASET_PATH)

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == ["label", "message"]
    assert len(dataframe) == 5574


def test_clean_dataset_columns():
    dataframe = load_raw_dataset(DATASET_PATH)
    cleaned = clean_dataset(dataframe)

    assert list(cleaned.columns) == ["label", "message"]


def test_clean_dataset_labels():
    dataframe = load_raw_dataset(DATASET_PATH)
    cleaned = clean_dataset(dataframe)

    assert set(cleaned["label"].unique()) == {"ham", "spam"}


def test_clean_dataset_removes_empty_values():
    dataframe = load_raw_dataset(DATASET_PATH)
    cleaned = clean_dataset(dataframe)

    assert not cleaned["label"].str.strip().eq("").any()
    assert not cleaned["message"].str.strip().eq("").any()
    assert cleaned["label"].notna().all()
    assert cleaned["message"].notna().all()


def test_clean_dataset_removes_duplicates():
    dataframe = load_raw_dataset(DATASET_PATH)
    cleaned = clean_dataset(dataframe)

    duplicate_count = cleaned.duplicated(
        subset=["label", "message"]
    ).sum()

    assert duplicate_count == 0


def test_clean_dataset_expected_size():
    dataframe = load_raw_dataset(DATASET_PATH)
    cleaned = clean_dataset(dataframe)

    assert len(cleaned) == 5160


def test_clean_dataset_class_distribution():
    dataframe = load_raw_dataset(DATASET_PATH)
    cleaned = clean_dataset(dataframe)

    class_counts = cleaned["label"].value_counts()

    assert class_counts["ham"] == 4518
    assert class_counts["spam"] == 642


def test_load_and_clean_dataset():
    dataframe = load_and_clean_dataset(DATASET_PATH)

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 5160


def test_clean_dataset_rejects_invalid_labels():
    dataframe = pd.DataFrame(
        {
            "label": ["ham", "spam", "unknown"],
            "message": [
                "Hello",
                "You won a prize!",
                "Test message",
            ],
        }
    )

    with pytest.raises(ValueError, match="Unexpected labels"):
        clean_dataset(dataframe)