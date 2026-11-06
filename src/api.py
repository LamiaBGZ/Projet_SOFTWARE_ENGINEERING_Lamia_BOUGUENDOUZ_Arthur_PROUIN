
import os
import secrets
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
ENV_PATH = PROJECT_ROOT / ".env"

# Charger les variables d'environnement
load_dotenv(ENV_PATH)

# Créer l'application FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API de prédiction du départ des clients télécoms",
    version="1.0.0",
)

# Configurer l'authentification par clé API
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """Vérifie la clé API envoyée dans la requête."""
    expected_key = os.getenv("API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="La clé API du serveur n'est pas configurée.",
        )

    if api_key is None or not secrets.compare_digest(
        api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé API absente ou invalide.",
        )

    return api_key


class CustomerData(BaseModel):
    """Données d'un client nécessaires à la prédiction."""

    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health_check() -> dict[str, str]:
    """Vérifie que l'API est disponible."""
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(
    customer: CustomerData,
    api_key: str = Security(verify_api_key),
) -> dict[str, Any]:
    """Prédit le départ d'un client avec le modèle entraîné."""

    # Vérifier que le modèle existe
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Le modèle entraîné est introuvable.",
        )

    # Charger le modèle
    model = joblib.load(MODEL_PATH)

    # Transformer les données reçues en DataFrame
    customer_df = pd.DataFrame([customer.model_dump()])

    # Prédire le départ du client
    prediction = int(model.predict(customer_df)[0])

    # Calculer la probabilité de départ
    probability = float(
        model.predict_proba(customer_df)[0][1]
    )

    # Retourner la réponse JSON
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "message": (
            "Le client risque de partir."
            if prediction == 1
            else "Le client devrait rester."
        ),
    }
