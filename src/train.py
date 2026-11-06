
from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_data
from src.preprocessing import clean_data


def train_model() -> None:
    """Entraîne, évalue et sauvegarde le modèle de prédiction du churn."""

    # 1. Charger et nettoyer les données
    data = clean_data(load_data())

    # 2. Séparer les variables explicatives et la cible
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    # 3. Identifier les colonnes numériques et catégorielles
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    # 4. Préparer les données
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    # 5. Créer le pipeline de Machine Learning
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # 6. Séparer les données : 80 % entraînement, 20 % test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 7. Entraîner le modèle
    model.fit(X_train, y_train)

    # 8. Évaluer les performances
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Clients pour l'entraînement :", len(X_train))
    print("Clients pour le test :", len(X_test))
    print(f"Précision (accuracy) : {accuracy:.2%}")

    # 9. Sauvegarder le modèle entraîné
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "churn_model.joblib"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print("Modèle sauvegardé :", model_path)


if __name__ == "__main__":
    train_model()
