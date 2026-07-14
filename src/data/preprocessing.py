import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils.config import RANDOM_STATE
def preprocess_dataset(df, target):

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    df = df.drop_duplicates()
    X = df.drop(columns=[target])
    y = df[target]
    categorical = X.select_dtypes(include=["object"]).columns
    for col in categorical:
        X[col] = LabelEncoder().fit_transform(
            X[col].astype(str)
        )
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print("Before SMOTE")
    print(y_train.value_counts())
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )
    print()
    print("After SMOTE")
    print(y_train.value_counts())
    return (
        X_train,
        X_test,
        y_train,
        y_test
    )