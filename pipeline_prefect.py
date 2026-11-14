from pathlib import Path
from prefect import task, flow
import subprocess
import sys
import argparse

from prepare import prepare
from train import train
from save_model import save_model
from load_model import load_model
from evaluate import evaluate

PROJECT_DIR = Path(__file__).resolve().parent

CODE_FILES = [
    PROJECT_DIR / name
    for name in [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py"
    ]
    if (PROJECT_DIR / name).is_file()
]

DATA_PATH = PROJECT_DIR / "Churn_Modelling.csv"
MODEL_PATH = PROJECT_DIR / "classifier.joblib"
@task
def install_dependencies():
    print("Installation des dependances...")
    subprocess.check_call([
        sys.executable, "-m", "pip",
        "install", "-r", "requirements.txt"
    ])
@task
def data_preparation():
    print("Preparation des donnees...")
    return prepare(DATA_PATH)
@task
def model_training(data):
    X_train, X_test, y_train, y_test, preprocessor = data
    print("Entrainement du modele...")
    model = train(X_train, y_train, preprocessor)
    return model, X_test, y_test
@task
def model_saving(model):
    print("Sauvegarde du modele...")
    save_model(model, MODEL_PATH)
@task
def model_loading():
    print("Chargement du modele...")
    return load_model(MODEL_PATH)
@task
def model_evaluation(model, X_test, y_test):
    print("Evaluation du modele...")
    evaluate(model, X_test, y_test)


@flow(name="Churn Pipeline")
def churn_pipeline():

    install_dependencies()
    data = data_preparation()
    model, X_test, y_test = model_training(data)
    model_saving(model)
    loaded_model = model_loading()
    model_evaluation(loaded_model, X_test, y_test)

@flow(name="Pipeline complet")
def flow_all():
    install_dependencies()
    data = data_preparation()
    model, X_test, y_test = model_training(data)
    model_saving(model)
    loaded_model = model_loading()
    model_evaluation(loaded_model, X_test, y_test)



@flow(name="Flow entrainement")
def flow_entrainement():
    install_dependencies()
    data = data_preparation()
    model, X_test, y_test = model_training(data)
    model_saving(model)


@flow(name="Flow evaluate")
def flow_evaluate():
    loaded_model = model_loading()

    data = data_preparation()
    X_train, X_test, y_train, y_test, preprocessor = data

    model_evaluation(loaded_model, X_test, y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gestion des flows Prefect"
    )

    parser.add_argument(
        "--flow",
        choices=["all", "entrainement", "evaluate"],
        default="all",
        help="Flow à exécuter"
    )

    args = parser.parse_args()

    if args.flow == "all":
        flow_all()

    elif args.flow == "entrainement":
        flow_entrainement()

    elif args.flow == "evaluate":
        flow_evaluate()

