import numpy as np

from src.feature_engineering import create_tfidf_vectorizer


def test_create_tfidf_vectorizer():
    vectorizer = create_tfidf_vectorizer()

    assert vectorizer is not None
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.max_df == 0.95
    assert vectorizer.sublinear_tf is True
    assert vectorizer.strip_accents == "unicode"


def test_tfidf_vectorizer_learns_vocabulary():
    vectorizer = create_tfidf_vectorizer()

    messages = [
        "hello how are you",
        "hello see you tomorrow",
        "win a free prize now",
        "free prize congratulations",
    ]

    transformed = vectorizer.fit_transform(messages)

    assert transformed.shape[0] == len(messages)
    assert transformed.shape[1] > 0
    assert len(vectorizer.vocabulary_) > 0


def test_tfidf_output_is_numeric():
    vectorizer = create_tfidf_vectorizer()

    messages = [
        "normal message for testing",
        "another normal message",
        "free prize congratulations",
        "free prize winner",
    ]

    transformed = vectorizer.fit_transform(messages)

    assert isinstance(transformed.toarray(), np.ndarray)
    assert np.isfinite(transformed.toarray()).all()


def test_tfidf_transform_after_fitting():
    vectorizer = create_tfidf_vectorizer()

    training_messages = [
        "hello how are you",
        "hello please call me",
        "please call me tomorrow",
        "free prize winner",
        "free prize congratulations",
        "claim your free reward",
    ]

    vectorizer.fit(training_messages)

    new_messages = [
        "hello please call me",
        "free reward winner",
    ]

    transformed = vectorizer.transform(new_messages)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] == len(vectorizer.vocabulary_)