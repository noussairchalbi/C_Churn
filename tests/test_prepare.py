
import pandas as pd
from prepare import prepare


def test_prepare(tmp_path):

    data = {
        "Geography": ["France", "Spain"] * 10,
        "Age": list(range(20, 40)),
        "Exited": [0, 1] * 10
    }

    df = pd.DataFrame(data)

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    X_train, X_test, y_train, y_test, preprocessor = prepare(
        csv_path
    )

    assert len(X_train) == 16
    assert len(X_test) == 4

    assert len(y_train) == 16
    assert len(y_test) == 4

    assert preprocessor is not None
    assert "Exited" not in X_train.columns
