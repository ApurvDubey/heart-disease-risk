"""Shared data loading and preprocessing for the heart disease risk project."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_2020_cleaned.csv"
TARGET = "HeartDisease"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split on the natural class balance. Done FIRST."""
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categoricals.

    One-hot rather than LabelEncoder: label encoding imposes a fake order
    on nominal columns such as Race.
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), cat_cols),
        ]
    )
