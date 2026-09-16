from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_and_clean_dataset
from src.feature_engineering import create_tfidf_vectorizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
MODEL_PATH = PROJECT_ROOT / "outputs" / "spam_classifier.joblib"


def prepare_data():
    """
    Load the raw dataset and prepare train/test splits.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    dataframe = load_and_clean_dataset(DATASET_PATH)

    X = dataframe["message"]
    y = dataframe["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def create_model_pipeline() -> Pipeline:
    """
    Create the complete TF-IDF + Logistic Regression pipeline.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing feature extraction
        and classification.
    """

    pipeline = Pipeline(
        steps=[
            ("tfidf", create_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return pipeline


def train_and_save_model():
    """
    Train the spam classifier and save the complete pipeline.
    """

    X_train, X_test, y_train, y_test = prepare_data()

    pipeline = create_model_pipeline()

    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    print("Model training completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Model saved to: {MODEL_PATH}")

    return pipeline, X_test, y_test


if __name__ == "__main__":
    train_and_save_model()