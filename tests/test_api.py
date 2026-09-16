from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Spam SMS/Email Classifier API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"
    assert data["model_loaded"] is True


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_spam_message():
    message = (
        "Congratulations! You have won a free prize. "
        "Call now to claim it!"
    )

    response = client.post(
        "/predict",
        json={"message": message},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_label"] in {"ham", "spam"}

    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["ham_probability"] <= 1
    assert 0 <= data["spam_probability"] <= 1

    assert (
        abs(
            data["ham_probability"]
            + data["spam_probability"]
            - 1
        )
        < 1e-6
    )


def test_predict_ham_message():
    message = "Hey, are we still meeting at 5 today?"

    response = client.post(
        "/predict",
        json={"message": message},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_label"] in {"ham", "spam"}

    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["ham_probability"] <= 1
    assert 0 <= data["spam_probability"] <= 1


def test_predict_empty_message():
    response = client.post(
        "/predict",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_predict_whitespace_message():
    response = client.post(
        "/predict",
        json={"message": "   "},
    )

    assert response.status_code == 422


def test_predict_missing_message():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_predict_message_too_long():
    message = "a" * 5001

    response = client.post(
        "/predict",
        json={"message": message},
    )

    assert response.status_code == 422