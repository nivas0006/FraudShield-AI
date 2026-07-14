from src.data.loader import load_dataset
from src.data.preprocessing import preprocess_dataset

print("="*60)
print("PaySim")
print("="*60)

paysim = load_dataset("paysim")

X_train,X_test,y_train,y_test = preprocess_dataset(
    paysim,
    target="isFraud"
)

print(X_train.shape)
print(X_test.shape)

print()

print("="*60)
print("Credit Card")
print("="*60)

credit = load_dataset("creditcard")

X_train,X_test,y_train,y_test = preprocess_dataset(
    credit,
    target="Class"
)
print(X_train.shape)
print(X_test.shape)