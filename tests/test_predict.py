
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.predict import predict_churn


def test_predict_churn_returns_integer() -> None:
    """Vérifie que la prédiction retourne un entier."""

    customer = pd.DataFrame({"tenure": [12]})

    with patch("src.predict.joblib.load") as mock_load:
        mock_load.return_value.predict.return_value = np.array([1])

        result = predict_churn(customer)

    assert isinstance(result, int)


def test_predict_churn_returns_zero() -> None:
    """Vérifie la prédiction d'un client qui reste."""

    customer = pd.DataFrame({"tenure": [24]})

    with patch("src.predict.joblib.load") as mock_load:
        mock_load.return_value.predict.return_value = np.array([0])

        result = predict_churn(customer)

    assert result == 0


def test_predict_churn_returns_one() -> None:
    """Vérifie la prédiction d'un client qui part."""

    customer = pd.DataFrame({"tenure": [1]})

    with patch("src.predict.joblib.load") as mock_load:
        mock_load.return_value.predict.return_value = np.array([1])

        result = predict_churn(customer)

    assert result == 1
