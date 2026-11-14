from pathlib import Path
import joblib


def load_model(file_path):
    """Charge un modèle sauvegardé avec joblib."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {file_path}")

    model = joblib.load(file_path)
    print(f"Modèle chargé : {file_path}")
    return model
