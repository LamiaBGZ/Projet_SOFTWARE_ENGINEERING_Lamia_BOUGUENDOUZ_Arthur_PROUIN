
import pandas as pd

from src.data_loader import load_data


def test_load_data_returns_dataframe() -> None:
    """Vérifie que le chargement retourne un DataFrame."""
    df = load_data()

    assert isinstance(df, pd.DataFrame)


def test_load_data_contains_expected_columns() -> None:
    """Vérifie la présence des colonnes importantes."""
    df = load_data()

    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert "TotalCharges" in df.columns


def test_load_data_has_expected_shape() -> None:
    """Vérifie les dimensions du dataset d'origine."""
    df = load_data()

    assert df.shape == (7043, 21)
