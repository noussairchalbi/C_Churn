from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def prepare(file_path, test_size=0.2, random_state=42):
    """Charge le CSV, sépare X/y et prépare l'encodage des variables catégorielles."""
    df = pd.read_csv(file_path)

    if "Exited" not in df.columns:
        raise ValueError("La colonne cible 'Exited' est absente du fichier CSV.")

    # Ces colonnes sont des identifiants ou des informations nominatives.
    df = df.drop(
        columns=[c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    )

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    numerical_columns = X.select_dtypes(exclude=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numerical", "passthrough", numerical_columns),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Nombre de lignes : {len(df)}")
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")

    return X_train, X_test, y_train, y_test, preprocessor
