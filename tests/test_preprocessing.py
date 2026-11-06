
import pandas as pd

from src.preprocessing import clean_data


def test_clean_data_removes_customer_id() -> None:
    """Vérifie que l'identifiant client est supprimé."""
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "TotalCharges": ["100.50", "200.75"],
        "Churn": ["Yes", "No"],
    })

    cleaned_df = clean_data(df)

    assert "customerID" not in cleaned_df.columns


def test_clean_data_converts_total_charges() -> None:
    """Vérifie que TotalCharges devient numérique."""
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "TotalCharges": ["100.50", "200.75"],
        "Churn": ["Yes", "No"],
    })

    cleaned_df = clean_data(df)

    assert pd.api.types.is_numeric_dtype(
        cleaned_df["TotalCharges"]
    )


def test_clean_data_encodes_churn() -> None:
    """Vérifie que Churn est encodé en 0 et 1."""
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "TotalCharges": ["100.50", "200.75"],
        "Churn": ["Yes", "No"],
    })

    cleaned_df = clean_data(df)

    assert cleaned_df["Churn"].tolist() == [1, 0]


def test_clean_data_removes_missing_values() -> None:
    """Vérifie que les lignes sans TotalCharges sont supprimées."""
    df = pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "TotalCharges": ["100.50", " ", "200.75"],
        "Churn": ["Yes", "No", "No"],
    })

    cleaned_df = clean_data(df)

    assert len(cleaned_df) == 2
    assert cleaned_df.isna().sum().sum() == 0
