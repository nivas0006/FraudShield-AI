from lightgbm import LGBMClassifier
def build_model():
    return LGBMClassifier(
        random_state=42
    )