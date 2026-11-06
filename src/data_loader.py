
from pathlib import Path

import pandas as pd


def load_data() -> pd.DataFrame:
    """Charge le dataset des clients télécoms."""
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "Telco-Customer-Churn.csv"

    data = pd.read_csv(csv_path)
    return data


if __name__ == "__main__":
    df = load_data()

    print("Dimensions du dataset :", df.shape)
    print("\nLes 5 premières lignes :")
    print(df.head())
