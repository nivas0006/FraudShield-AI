import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
def train_model(model,
                X_train,
                y_train,
                X_test,
                y_test):
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":
        accuracy_score(
            y_test,
            predictions
        ),
        "Precision":
        precision_score(
            y_test,
            predictions
        ),
        "Recall":
        recall_score(
            y_test,
            predictions
        ),
        "F1":
        f1_score(
            y_test,
            predictions
        ),
        "ROC_AUC":
        roc_auc_score(
            y_test,
            probabilities
        ),
        "Training_Time":
        training_time
    }
    return metrics