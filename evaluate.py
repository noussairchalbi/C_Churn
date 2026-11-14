from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt


def evaluate(rf, x_test, y_test):

    y_pred = rf.predict(x_test)

    # Evaluation of model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy score:", accuracy * 100)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.show()

    return y_pred
