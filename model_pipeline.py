from pathlib import Path
from prepare import prepare
from train import train
from evaluate import evaluate
from save_model import save_model

# Le CSV doit se trouver dans ~/ml_project/
PROJECT_DIR = Path.home() / "ml_project"
DATA_PATH = PROJECT_DIR / "Churn_Modelling.csv"
MODEL_PATH = PROJECT_DIR / "classifier.joblib"


def main():
    # 1. Préparer les données
    X_train, X_test, y_train, y_test, preprocessor = prepare(DATA_PATH)

    # 2. Entraîner le modèle
    model = train(X_train, y_train, preprocessor)

    # 3. Évaluer le modèle
    evaluate(model, X_test, y_test)
	

    save_model(model, MODEL_PATH)
if __name__ == "__main__":
    main()
