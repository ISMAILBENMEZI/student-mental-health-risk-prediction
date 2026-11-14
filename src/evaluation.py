import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.config import METRICS_PATH, CONFUSION_MATRIX_PATH


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(
            y_test, y_proba, average=("weighted"), multi_class="ovr"
        ),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    for metric, value in metrics.items():
        if metric != "classification_report":
            print(f"{metric} : {value:.4f}")

        print("\n classification_report: ")
        print(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=model.classes_, cmap="Blues", ax=ax
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)
