from pathlib import Path

import pandas as pd


EXPECTED_LABELS = {"ham", "spam"}


def load_raw_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load the UCI SMS Spam Collection dataset.

    The raw dataset is tab-separated and has no header:
        label <TAB> message

    Parameters
    ----------
    file_path : str | Path
        Path to the SMSSpamCollection file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - label
        - message

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the dataset format or labels are invalid.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {file_path}"
        )

    try:
        dataframe = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8",
            quoting=3,
        )
    except UnicodeDecodeError as error:
        raise ValueError(
            "The dataset could not be decoded as UTF-8."
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "The dataset could not be parsed as a tab-separated file."
        ) from error

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    return dataframe


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the SMS spam dataset.

    Cleaning operations:
    1. Validate required columns.
    2. Remove rows with missing labels/messages.
    3. Normalize labels and messages.
    4. Validate that labels are ham or spam.
    5. Remove exact duplicate label-message pairs.
    6. Reset the DataFrame index.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw loaded dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    required_columns = {"label", "message"}

    if not required_columns.issubset(dataframe.columns):
        missing_columns = required_columns - set(dataframe.columns)
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    cleaned = dataframe[["label", "message"]].copy()

    # Remove missing values.
    cleaned = cleaned.dropna(subset=["label", "message"])

    # Convert values to strings and remove unnecessary surrounding whitespace.
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned["message"] = cleaned["message"].astype(str).str.strip()

    # Remove rows with empty labels or messages.
    cleaned = cleaned[
        (cleaned["label"] != "")
        & (cleaned["message"] != "")
    ]

    # Validate class labels.
    actual_labels = set(cleaned["label"].unique())
    invalid_labels = actual_labels - EXPECTED_LABELS

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels found in dataset: {sorted(invalid_labels)}"
        )

    # Remove exact duplicate rows.
    cleaned = cleaned.drop_duplicates(
        subset=["label", "message"],
        keep="first",
    )

    # Reset the index after cleaning.
    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    return cleaned


def load_and_clean_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load and clean the SMS spam dataset in one operation.

    Parameters
    ----------
    file_path : str | Path
        Path to the SMSSpamCollection file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    dataframe = load_raw_dataset(file_path)
    return clean_dataset(dataframe)


if __name__ == "__main__":
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "SMSSpamCollection"
    )

    dataframe = load_and_clean_dataset(dataset_path)

    print("Dataset loaded successfully.")
    print(f"Rows after cleaning: {len(dataframe)}")
    print("\nClass distribution:")
    print(dataframe["label"].value_counts())

    print("\nFirst 5 rows:")
    print(dataframe.head())