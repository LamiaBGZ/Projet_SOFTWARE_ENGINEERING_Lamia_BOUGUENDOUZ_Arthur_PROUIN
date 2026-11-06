
import os

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_check() -> None:
    """Vérifie que la route /health fonctionne."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_without_api_key() -> None:
    """Vérifie que /predict refuse une requête sans clé API."""
    response = client.post("/predict", json={})

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Clé API absente ou invalide."
    }


def test_predict_with_invalid_api_key() -> None:
    """Vérifie que /predict refuse une mauvaise clé API."""
    response = client.post(
        "/predict",
        json={},
        headers={"X-API-Key": "mauvaise-cle"},
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Clé API absente ou invalide."
    }


def test_predict_with_valid_api_key() -> None:
    """Vérifie qu'une requête autorisée retourne une prédiction."""
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    api_key = os.getenv("API_KEY")
    assert api_key is not None, "La clé API doit être configurée."

    response = client.post(
        "/predict",
        json=customer,
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert isinstance(result["message"], str)
