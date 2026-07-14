import pandas as pd
from src.data.loader import load_dataset
from src.data.preprocessing import preprocess_dataset
from src.models.logistic import build_model as logistic
from src.models.decision_tree import build_model as decision_tree
from src.models.random_forest import build_model as random_forest
from src.models.xgboost_model import build_model as xgboost
from src.models.lightgbm_model import build_model as lightgbm
from src.models.trainer import train_model

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
print("\n")
print("=" * 80)
print("MODEL LEADERBOARD")
print("=" * 80)
print(leaderboard)

leaderboard.to_csv(
    "outputs/model_leaderboard.csv",
    index=False
)