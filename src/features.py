# src/features.py
"""Feature engineering and preprocessing pipelines for Titanic."""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Final numeric columns after cleaning
NUMERIC_COLS: List[str] = [
    "Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketGroupSize"
]

# Binary columns are already 0/1; pass-through is fine
BINARY_COLS: List[str] = [
    "IsAlone", "HasCabin"
]

# Treat Pclass as categorical for a safe One-Hot
CATEGORICAL_COLS: List[str] = [
    "Pclass", "Sex", "Embarked", "Deck", "Title", "TicketPrefix"
]

# Columns not used by the model
DROP_COLS: List[str] = [
    "Survived",     # target
    "PassengerId"   # identifier
]


def get_feature_groups(df: pd.DataFrame) -> Tuple[list, list, list]:
    """
    Return the three feature groups.
    Currently static, but kept as a function for easy future tweaks.
    """
    return NUMERIC_COLS, BINARY_COLS, CATEGORICAL_COLS


def build_preprocess_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    ColumnTransformer that:
      - imputes numerics with median, then standardizes
      - passes binary 0/1 as-is
      - imputes categoricals with most_frequent, then OneHot-encodes
    """

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    binary_pipeline = "passthrough"  # values are already 0/1


    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    num_cols, bin_cols, cat_cols = get_feature_groups(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("bin", binary_pipeline, bin_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor
