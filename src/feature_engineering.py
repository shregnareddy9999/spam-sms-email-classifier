from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create the TF-IDF vectorizer used by the spam classifier.

    Returns
    -------
    TfidfVectorizer
        Configured TF-IDF vectorizer.
    """

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )