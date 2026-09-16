from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "spam_classifier.joblib"
EXAMPLES_PATH = PROJECT_ROOT / "outputs" / "example_predictions.csv"


def load_model():
    """
    Load the trained spam classification pipeline.

    Returns
    -------
    object
        Trained scikit-learn pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model was not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_messages(messages: list[str]) -> pd.DataFrame:
    """
    Predict spam/ham labels and probabilities for messages.

    Parameters
    ----------
    messages : list[str]
        Messages to classify.

    Returns
    -------
    pd.DataFrame
        Prediction results containing:
        - message
        - predicted_label
        - ham_probability
        - spam_probability
        - confidence
    """

    if not messages:
        raise ValueError("At least one message is required.")

    if any(not isinstance(message, str) for message in messages):
        raise TypeError("Every message must be a string.")

    if any(not message.strip() for message in messages):
        raise ValueError("Messages cannot be empty.")

    model = load_model()

    predictions = model.predict(messages)
    probabilities = model.predict_proba(messages)

    class_names = list(model.classes_)

    ham_index = class_names.index("ham")
    spam_index = class_names.index("spam")

    results = []

    for message, prediction, probability in zip(
        messages,
        predictions,
        probabilities,
    ):
        ham_probability = float(probability[ham_index])
        spam_probability = float(probability[spam_index])

        confidence = max(
            ham_probability,
            spam_probability,
        )

        results.append(
            {
                "message": message,
                "predicted_label": prediction,
                "ham_probability": ham_probability,
                "spam_probability": spam_probability,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(results)


def generate_example_predictions() -> pd.DataFrame:
    """
    Generate the required example predictions.

    Returns
    -------
    pd.DataFrame
        Predictions for at least 10 example messages.
    """

    example_messages = [
        "Congratulations! You have won a free prize. Call now to claim it!",
        "Hey, are we still meeting at 5 today?",
        "URGENT! You have been selected for a cash reward. Reply WIN now!",
        "Can you send me the notes from today's class?",
        "Free entry in a weekly competition to win a brand new mobile phone!",
        "I'll call you when I reach home.",
        "You have won £1000! Text WIN to 87121 to claim your prize.",
        "Don't forget to bring your project tomorrow.",
        "Claim your exclusive free gift now by calling this number.",
        "Are you coming to the library after lunch?",
        "Congratulations! Your mobile number has won a cash prize.",
        "Please let me know when you are free.",
    ]

    results = predict_messages(example_messages)

    results["confidence_percentage"] = (
        results["confidence"] * 100
    ).round(2)

    results["ham_probability_percentage"] = (
        results["ham_probability"] * 100
    ).round(2)

    results["spam_probability_percentage"] = (
        results["spam_probability"] * 100
    ).round(2)

    results.to_csv(
        EXAMPLES_PATH,
        index=False,
    )

    return results


if __name__ == "__main__":
    results = generate_example_predictions()

    print("Example predictions generated successfully.")
    print(f"Saved to: {EXAMPLES_PATH}")
    print()
    print(
        results[
            [
                "message",
                "predicted_label",
                "ham_probability_percentage",
                "spam_probability_percentage",
                "confidence_percentage",
            ]
        ].to_string(index=False)
    )