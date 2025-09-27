"""CLI training script for Titanic model (headless).

Reads a cleaned training CSV, builds preprocessing + model pipeline,
fits with class-weighted sample weights, and saves the trained pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.features import build_preprocess_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Titanic survival model")
    p.add_argument(
        "--train-path",
        default="data/processed/train_clean.csv",
        help="Path to cleaned training CSV with Survived column",
    )
    p.add_argument(
        "--model-out",
        default="models/best_gb_pipeline.joblib",
        help="Where to save the trained pipeline",
    )
    p.add_argument("--n-estimators", type=int, default=1141)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument(
        "--class-weight-0", type=float, default=0.81, help="Weight for class 0 (non-survivors)"
    )
    p.add_argument(
        "--class-weight-1", type=float, default=1.30, help="Weight for class 1 (survivors)"
    )
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    df = pd.read_csv(train_path)
    if "Survived" not in df.columns:
        raise ValueError("Training CSV must contain 'Survived' column")

    y = df["Survived"].astype(int)

    pre = build_preprocess_pipeline(df)
    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        random_state=args.random_state,
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model),
    ])

    # Convert class weights to per-sample weights
    weight_map = {0: float(args.class_weight_0), 1: float(args.class_weight_1)}
    sample_weight = y.map(weight_map).values

    pipe.fit(df, y, model__sample_weight=sample_weight)

    out_path = Path(args.model_out)
    os.makedirs(out_path.parent, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"Saved trained pipeline to: {out_path}")


if __name__ == "__main__":
    main()

