from pathlib import Path

import joblib

from src.train_model import (
    MODEL_PATH,
    create_model_pipeline,
    prepare_data,
)


def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data()

    assert len(X_train) == 4128
    assert len(X_test) == 1032

    assert len(y_train) == 4128
    assert len(y_test) == 1032

    assert len(X_train) + len(X_test) == 5160
    assert len(y_train) + len(y_test) == 5160


def test_prepare_data_contains_both_classes():
    _, _, y_train, y_test = prepare_data()

    assert set(y_train.unique()) == {"ham", "spam"}
    assert set(y_test.unique()) == {"ham", "spam"}


def test_create_model_pipeline():
    pipeline = create_model_pipeline()

    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    assert pipeline.named_steps["tfidf"] is not None
    assert pipeline.named_steps["classifier"] is not None

    classifier = pipeline.named_steps["classifier"]

    assert classifier.max_iter == 1000
    assert classifier.class_weight == "balanced"


def test_model_can_train_and_predict():
    X_train, X_test, y_train, _ = prepare_data()

    pipeline = create_model_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({"ham", "spam"})


def test_model_outputs_probabilities():
    X_train, X_test, y_train, _ = prepare_data()

    pipeline = create_model_pipeline()

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)

    assert probabilities.shape[0] == len(X_test)
    assert probabilities.shape[1] == 2

    for probability_pair in probabilities:
        assert abs(probability_pair.sum() - 1.0) < 1e-6


def test_saved_model_exists():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.is_file()


def test_saved_model_can_be_loaded():
    assert MODEL_PATH.exists()

    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert "tfidf" in model.named_steps
    assert "classifier" in model.named_steps


def test_saved_model_can_predict():
    assert MODEL_PATH.exists()

    model = joblib.load(MODEL_PATH)

    messages = [
        "Congratulations! You won a free prize!",
        "Are we still meeting today?",
    ]

    predictions = model.predict(messages)

    assert len(predictions) == 2
    assert set(predictions).issubset({"ham", "spam"})


def test_model_path_is_inside_outputs_directory():
    assert MODEL_PATH.parent.name == "outputs"
    assert MODEL_PATH.name == "spam_classifier.joblib"
    assert isinstance(MODEL_PATH, Path)