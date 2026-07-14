import pandas as pd
from src.data.loader import load_dataset
from src.data.preprocessing import preprocess_dataset
from src.models.logistic import build_model as logistic
from src.models.decision_tree import build_model as decision_tree
from src.models.random_forest import build_model as random_forest
from src.models.xgboost_model import build_model as xgboost
from src.models.save_model import save_model
from src.models.lightgbm_model import build_model as lightgbm
from src.models.trainer import train_model
from src.evaluation.plots import (
    save_confusion_matrix,
    save_roc_curve,
    save_precision_recall
)

credit = load_dataset("creditcard")

X_train, X_test, y_train, y_test = preprocess_dataset(
    credit,
    target="Class"
)

models = {
    "Logistic Regression": logistic(),
    "Decision Tree": decision_tree(),
    "Random Forest": random_forest(),
    "XGBoost": xgboost(),
    "LightGBM": lightgbm()
}
results = []

for name, model in models.items():
    print("=" * 60)
    print(name)
    print("=" * 60)

    metrics = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )
    metrics["Model"] = name
    results.append(metrics)
leaderboard = pd.DataFrame(results)
leaderboard = leaderboard.sort_values(
    "ROC_AUC",
    ascending=False
)
best_model_name = leaderboard.iloc[0]["Model"]

print("\nBest Model:", best_model_name)
trained_models = {
    "Logistic Regression": models["Logistic Regression"],
    "Decision Tree": models["Decision Tree"],
    "Random Forest": models["Random Forest"],
    "XGBoost": models["XGBoost"],
    "LightGBM": models["LightGBM"],
}

save_model(
    trained_models[best_model_name],
    "best_model"
)
best_model = trained_models[best_model_name]

save_confusion_matrix(
    best_model,
    X_test,
    y_test
)

save_roc_curve(
    best_model,
    X_test,
    y_test
)

save_precision_recall(
    best_model,
    X_test,
    y_test
)
leaderboard = pd.DataFrame(results)
leaderboard = leaderboard.sort_values(
    "ROC_AUC",
    ascending=False
)
print("\n")
print("=" * 80)
print("MODEL LEADERBOARD")
print("=" * 80)
print(leaderboard)

leaderboard.to_csv(
    "outputs/model_leaderboard.csv",
    index=False
)