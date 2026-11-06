
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import clean_data


def predict_churn(customer: pd.DataFrame) -> int:
    """Prédit si un client risque de quitter l'opérateur."""

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "churn_model.joblib"

    # Charger le modèle déjà entraîné
    model = joblib.load(model_path)

    # Effectuer la prédiction
    prediction = model.predict(customer)

    return int(prediction[0])


if __name__ == "__main__":
    # Utiliser un client du dataset pour tester la prédiction
    data = clean_data(load_data())

    customer = data.drop(columns=["Churn"]).iloc[[0]]

    result = predict_churn(customer)

    if result == 1:
        print("Prédiction : le client risque de partir.")
    else:
        print("Prédiction : le client devrait rester.")
