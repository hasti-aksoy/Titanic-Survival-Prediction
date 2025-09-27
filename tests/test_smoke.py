import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.features import build_preprocess_pipeline


def _tiny_df(n: int = 3) -> pd.DataFrame:
    # Minimal rows with expected columns
    data = {
        "PassengerId": list(range(1, n + 1)),
        "Survived": [0, 1, 0][:n],
        "Pclass": [1, 2, 3][:n],
        "Sex": ["male", "female", "male"][:n],
        "Age": [22.0, np.nan, 35.0][:n],
        "SibSp": [1, 0, 0][:n],
        "Parch": [0, 1, 0][:n],
        "FamilySize": [2, 2, 1][:n],
        "IsAlone": [0, 0, 1][:n],
        "Fare": [7.25, 71.2833, 8.05][:n],
        "Embarked": ["S", "C", "S"][:n],
        "Deck": ["Unknown", "C", "Unknown"][:n],
        "HasCabin": [0, 1, 0][:n],
        "TicketGroupSize": [1, 1, 1][:n],
        "TicketPrefix": ["A", "B", "A"][:n],
        "Title": ["Mr", "Mrs", "Mr"][:n],
    }
    return pd.DataFrame(data)


def test_build_preprocess_pipeline_smoke():
    df = _tiny_df(3)
    pre = build_preprocess_pipeline(df)
    X = pre.fit_transform(df)
    assert X.shape[0] == len(df)
    assert not np.isnan(X).any()


def test_training_and_inference_pipeline_smoke():
    df = _tiny_df(3)
    y = df["Survived"].astype(int)

    pre = build_preprocess_pipeline(df)
    model = GradientBoostingClassifier(n_estimators=5, random_state=0)
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    pipe.fit(df, y)

    proba = pipe.predict_proba(df)[:, 1]
    assert proba.shape == (len(df),)
    assert np.isfinite(proba).all()

