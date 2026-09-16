from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
MODEL_PATH = PROJECT_ROOT / "outputs" / "spam_classifier.joblib"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

METRICS_PATH = OUTPUTS_DIR / "metrics.csv"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
METRICS_PLOT_PATH = PLOTS_DIR / "metrics.png"


def load_test_data():
    """
    Recreate the exact test split used during model training.

    Returns
    -------
    tuple
        X_test, y_test
    """

    from src.data_preprocessing import load_and_clean_dataset

    dataframe = load_and_clean_dataset(DATASET_PATH)

    X = dataframe["message"]
    y = dataframe["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


def evaluate_model():
    """
    Evaluate the saved spam classification pipeline.

    Saves:
    - evaluation metrics
    - test predictions and probabilities
    - confusion matrix
    - metrics visualization
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model was not found: {MODEL_PATH}"
        )

    X_test, y_test = load_test_data()

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    class_names = list(model.classes_)

    ham_index = class_names.index("ham")
    spam_index = class_names.index("spam")

    ham_probabilities = probabilities[:, ham_index]
    spam_probabilities = probabilities[:, spam_index]

    confidence = probabilities.max(axis=1)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    metrics = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ],
            "value": [
                accuracy,
                precision,
                recall,
                f1,
            ],
        }
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    # Save individual test predictions and probabilities.
    test_predictions = pd.DataFrame(
        {
            "message": X_test.to_numpy(),
            "actual_label": y_test.to_numpy(),
            "predicted_label": predictions,
            "ham_probability": ham_probabilities,
            "spam_probability": spam_probabilities,
            "confidence": confidence,
        }
    )

    test_predictions.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    # Create confusion matrix.
    labels = ["ham", "spam"]

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )

    plt.title("Spam Classifier Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # Create metrics chart.
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=metrics,
        x="metric",
        y="value",
    )

    plt.ylim(0, 1)
    plt.title("Spam Classifier Performance Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        METRICS_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("Model evaluation completed successfully.")
    print()
    print("Evaluation metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print()
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=["Actual HAM", "Actual SPAM"],
            columns=["Predicted HAM", "Predicted SPAM"],
        )
    )
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )
    print(f"Metrics saved to: {METRICS_PATH}")
    print(
        f"Test predictions saved to: "
        f"{TEST_PREDICTIONS_PATH}"
    )
    print(
        f"Confusion matrix saved to: "
        f"{CONFUSION_MATRIX_PATH}"
    )
    print(
        f"Metrics plot saved to: "
        f"{METRICS_PLOT_PATH}"
    )

    return metrics, matrix, test_predictions


if __name__ == "__main__":
    evaluate_model()