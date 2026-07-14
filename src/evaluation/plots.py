import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

os.makedirs("outputs", exist_ok=True)
def save_confusion_matrix(model, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )
    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def save_roc_curve(model, X_test, y_test):
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )
    plt.savefig(
        "outputs/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def save_precision_recall(model, X_test, y_test):
    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test
    )
    plt.savefig(
        "outputs/precision_recall.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()