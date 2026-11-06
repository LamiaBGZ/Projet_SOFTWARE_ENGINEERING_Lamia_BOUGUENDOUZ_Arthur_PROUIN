
import pandas as pd

from src.data_loader import load_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données des clients télécoms."""
    df = df.copy()

    # Convertir TotalCharges en nombres
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Supprimer les lignes avec TotalCharges manquant
    df = df.dropna(subset=["TotalCharges"])

    # Supprimer l'identifiant du client
    df = df.drop(columns=["customerID"])

    # Convertir la cible Churn en 0 et 1
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


if __name__ == "__main__":
    data = load_data()
    cleaned_data = clean_data(data)

    print("Avant nettoyage :", data.shape)
    print("Après nettoyage :", cleaned_data.shape)
    print("\nValeurs manquantes :", cleaned_data.isnull().sum().sum())
    print("\nRépartition de Churn :")
    print(cleaned_data["Churn"].value_counts())
