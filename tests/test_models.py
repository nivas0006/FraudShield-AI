from src.data.loader import load_dataset
from src.data.preprocessing import preprocess_dataset
from src.models.logistic import build_model
from src.models.trainer import train_model

credit = load_dataset("creditcard")
X_train, X_test, y_train, y_test = preprocess_dataset(
    credit,
    target="Class"
)
model = build_model()
results = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
)
print(results)