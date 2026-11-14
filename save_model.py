from pathlib import Path
import joblib


def save_model(model, file_path):
    """Sauvegarde le modèle entraîné avec joblib."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Modèle sauvegardé : {file_path}")
