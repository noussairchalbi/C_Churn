from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train(X_train, y_train, preprocessor, random_state=42):
    """Construit et entraîne le pipeline de prétraitement et Random Forest."""
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    print("Entraînement terminé.")
    return model
