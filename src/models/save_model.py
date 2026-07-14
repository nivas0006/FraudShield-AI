import joblib
import os
def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}.pkl"
    joblib.dump(model, path)
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)
    print(path)