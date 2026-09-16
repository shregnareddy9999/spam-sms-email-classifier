from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pandas as pd

from src.evaluate_model import (
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    METRICS_PLOT_PATH,
    TEST_PREDICTIONS_PATH,
    evaluate_model,
    load_test_data,
)


def test_load_test_data():
    X_test, y_test = load_test_data()

    assert len(X_test) == 1032
    assert len(y_test) == 1032

    assert set(y_test.unique()) == {"ham", "spam"}


def test_evaluate_model_returns_expected_results():
    metrics, matrix, test_predictions = evaluate_model()

    assert isinstance(metrics, pd.DataFrame)
    assert metrics.shape[0] == 4

    assert set(metrics["metric"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    assert matrix.shape == (2, 2)

    assert isinstance(test_predictions, pd.DataFrame)
    assert len(test_predictions) == 1032


def test_evaluation_metrics_are_valid():
    metrics, _, _ = evaluate_model()

    for value in metrics["value"]:
        assert 0 <= value <= 1


def test_confusion_matrix_contains_valid_counts():
    _, matrix, _ = evaluate_model()

    assert matrix.sum() == 1032
    assert (matrix >= 0).all()


def test_test_predictions_have_expected_columns():
    _, _, test_predictions = evaluate_model()

    expected_columns = {
        "message",
        "actual_label",
        "predicted_label",
        "ham_probability",
        "spam_probability",
        "confidence",
    }

    assert set(test_predictions.columns) == expected_columns


def test_test_predictions_have_valid_labels():
    _, _, test_predictions = evaluate_model()

    assert set(test_predictions["actual_label"]) == {
        "ham",
        "spam",
    }

    assert set(test_predictions["predicted_label"]).issubset(
        {"ham", "spam"}
    )


def test_prediction_probabilities_are_valid():
    _, _, test_predictions = evaluate_model()

    assert (
        test_predictions["ham_probability"]
        .between(0, 1)
        .all()
    )

    assert (
        test_predictions["spam_probability"]
        .between(0, 1)
        .all()
    )

    assert (
        test_predictions["confidence"]
        .between(0, 1)
        .all()
    )

    probability_sums = (
        test_predictions["ham_probability"]
        + test_predictions["spam_probability"]
    )

    assert (
        probability_sums.sub(1).abs() < 1e-6
    ).all()


def test_evaluation_files_are_created():
    evaluate_model()

    expected_files = [
        METRICS_PATH,
        TEST_PREDICTIONS_PATH,
        CONFUSION_MATRIX_PATH,
        METRICS_PLOT_PATH,
    ]

    for file_path in expected_files:
        assert file_path.exists()
        assert file_path.is_file()


def test_metrics_csv_can_be_loaded():
    evaluate_model()

    assert METRICS_PATH.exists()

    metrics = pd.read_csv(METRICS_PATH)

    assert len(metrics) == 4
    assert "metric" in metrics.columns
    assert "value" in metrics.columns


def test_test_predictions_csv_can_be_loaded():
    evaluate_model()

    assert TEST_PREDICTIONS_PATH.exists()

    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)

    assert len(predictions) == 1032
    assert "message" in predictions.columns
    assert "predicted_label" in predictions.columns


def test_output_plots_are_non_empty():
    evaluate_model()

    assert CONFUSION_MATRIX_PATH.stat().st_size > 0
    assert METRICS_PLOT_PATH.stat().st_size > 0


def test_evaluation_files_are_png_and_csv():
    evaluate_model()

    assert CONFUSION_MATRIX_PATH.suffix == ".png"
    assert METRICS_PLOT_PATH.suffix == ".png"
    assert METRICS_PATH.suffix == ".csv"
    assert TEST_PREDICTIONS_PATH.suffix == ".csv"


def test_output_paths_are_inside_expected_directories():
    assert METRICS_PATH.parent.name == "outputs"
    assert TEST_PREDICTIONS_PATH.parent.name == "outputs"
    assert CONFUSION_MATRIX_PATH.parent.name == "plots"
    assert METRICS_PLOT_PATH.parent.name == "plots"

    assert isinstance(METRICS_PATH, Path)