import pandas as pd
import pytest

from src.predict import (
    EXAMPLES_PATH,
    MODEL_PATH,
    generate_example_predictions,
    load_model,
    predict_messages,
)


def test_model_file_exists():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.is_file()


def test_load_model():
    model = load_model()

    assert model is not None
    assert "tfidf" in model.named_steps
    assert "classifier" in model.named_steps


def test_predict_single_message():
    results = predict_messages(
        ["Congratulations! You have won a free prize!"]
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1

    assert results.iloc[0]["predicted_label"] in {
        "ham",
        "spam",
    }


def test_predict_multiple_messages():
    messages = [
        "Congratulations! You have won a free prize!",
        "Hey, are we still meeting at 5 today?",
        "Please send me the notes from class.",
    ]

    results = predict_messages(messages)

    assert len(results) == len(messages)
    assert list(results["message"]) == messages

    assert set(results["predicted_label"]).issubset(
        {"ham", "spam"}
    )


def test_prediction_probabilities_are_valid():
    messages = [
        "Congratulations! You have won a free prize!",
        "Hey, are we still meeting at 5 today?",
    ]

    results = predict_messages(messages)

    assert (
        results["ham_probability"]
        .between(0, 1)
        .all()
    )

    assert (
        results["spam_probability"]
        .between(0, 1)
        .all()
    )

    assert (
        results["confidence"]
        .between(0, 1)
        .all()
    )

    probability_sums = (
        results["ham_probability"]
        + results["spam_probability"]
    )

    assert (
        probability_sums.sub(1).abs() < 1e-6
    ).all()


def test_confidence_matches_highest_probability():
    messages = [
        "Congratulations! You have won a free prize!",
        "Hey, are we still meeting at 5 today?",
    ]

    results = predict_messages(messages)

    expected_confidence = results[
        ["ham_probability", "spam_probability"]
    ].max(axis=1)

    assert (
        results["confidence"].reset_index(drop=True)
        == expected_confidence.reset_index(drop=True)
    ).all()


def test_predict_empty_list():
    with pytest.raises(
        ValueError,
        match="At least one message is required",
    ):
        predict_messages([])


def test_predict_non_string_message():
    with pytest.raises(
        TypeError,
        match="Every message must be a string",
    ):
        predict_messages(
            ["Hello", 123]
        )


def test_predict_empty_message():
    with pytest.raises(
        ValueError,
        match="Messages cannot be empty",
    ):
        predict_messages(
            ["Hello", "   "]
        )


def test_generate_example_predictions():
    results = generate_example_predictions()

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 12

    assert EXAMPLES_PATH.exists()
    assert EXAMPLES_PATH.is_file()

    assert (
        "message" in results.columns
    )

    assert (
        "predicted_label" in results.columns
    )

    assert (
        "confidence_percentage" in results.columns
    )


def test_example_predictions_have_valid_labels():
    results = generate_example_predictions()

    assert set(results["predicted_label"]).issubset(
        {"ham", "spam"}
    )


def test_example_predictions_have_valid_percentages():
    results = generate_example_predictions()

    assert (
        results["confidence_percentage"]
        .between(0, 100)
        .all()
    )

    assert (
        results["ham_probability_percentage"]
        .between(0, 100)
        .all()
    )

    assert (
        results["spam_probability_percentage"]
        .between(0, 100)
        .all()
    )


def test_example_prediction_probabilities_sum_to_100():
    results = generate_example_predictions()

    probability_sums = (
        results["ham_probability_percentage"]
        + results["spam_probability_percentage"]
    )

    assert (
        (probability_sums - 100).abs() < 0.01
    ).all()


def test_example_predictions_csv_can_be_loaded():
    generate_example_predictions()

    saved_results = pd.read_csv(EXAMPLES_PATH)

    assert len(saved_results) == 12

    assert (
        "predicted_label" in saved_results.columns
    )

    assert (
        "confidence_percentage" in saved_results.columns
    )