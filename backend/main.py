from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "spam_classifier.joblib"


app = FastAPI(
    title="Spam SMS/Email Classifier API",
    description=(
        "API for classifying text messages as HAM or SPAM "
        "using a TF-IDF and Logistic Regression pipeline."
    ),
    version="1.0.0",
)


# Allow the frontend to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """
    Request body for the prediction endpoint.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="SMS or email text to classify.",
    )


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction endpoint.
    """

    predicted_label: str
    confidence: float
    ham_probability: float
    spam_probability: float


def load_model():
    """
    Load the trained ML pipeline from disk.

    Returns
    -------
    object
        Trained scikit-learn pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    model = None


@app.get("/")
def root():
    """
    Return basic API information.
    """

    return {
        "name": "Spam SMS/Email Classifier API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    """
    Check API and model health.
    """

    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    """
    Classify a message as HAM or SPAM.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not available.",
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    class_names = list(model.classes_)

    ham_index = class_names.index("ham")
    spam_index = class_names.index("spam")

    ham_probability = float(
        probabilities[ham_index]
    )

    spam_probability = float(
        probabilities[spam_index]
    )

    confidence = max(
        ham_probability,
        spam_probability,
    )

    return PredictionResponse(
        predicted_label=str(prediction),
        confidence=confidence,
        ham_probability=ham_probability,
        spam_probability=spam_probability,
    )