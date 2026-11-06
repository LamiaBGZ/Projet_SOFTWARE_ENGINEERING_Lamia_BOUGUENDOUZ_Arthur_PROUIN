
from pathlib import Path
from unittest.mock import patch

import joblib
from sklearn.pipeline import Pipeline

from src.train import train_model


def test_train_model_saves_pipeline(tmp_path: Path) -> None:
    """Vérifie que l'entraînement sauvegarde un pipeline utilisable."""

    temporary_model_path = tmp_path / "churn_model.joblib"

    with patch("src.train.joblib.dump") as mock_dump:
        train_model()

    assert mock_dump.call_count == 1

    trained_model, saved_path = mock_dump.call_args.args

    assert isinstance(trained_model, Pipeline)
    assert saved_path.name == temporary_model_path.name
    assert "preprocessor" in trained_model.named_steps
    assert "classifier" in trained_model.named_steps
