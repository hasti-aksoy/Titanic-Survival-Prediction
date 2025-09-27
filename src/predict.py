"""CLI prediction script for Titanic model (headless).

Loads a saved pipeline, runs predict_proba on an input CSV, and writes
predicted labels (with optional probabilities) to a CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict Titanic survival using a saved pipeline")
    p.add_argument(
        "--input",
        default="data/processed/test_clean.csv",
        help="Path to cleaned input CSV (no Survived column)",
    )
    p.add_argument(
        "--model",
        default="models/best_gb_pipeline.joblib",
        help="Path to saved pipeline (.joblib)",
    )
    p.add_argument(
        "--output",
        default="submission.csv",
        help="Where to write predictions CSV",
    )
    p.add_argument(
        "--threshold", type=float, default=0.47, help="Decision threshold for class 1"
    )
    p.add_argument(
        "--write-proba",
        action="store_true",
        help="Include probability column in the output",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df = pd.read_csv(args.input)
    pipe = joblib.load(model_path)

    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= float(args.threshold)).astype(int)

    out = pd.DataFrame({"Survived": pred})
    if "PassengerId" in df.columns:
        out.insert(0, "PassengerId", df["PassengerId"].values)
    if args.write_proba:
        out["SurvivalProbability"] = proba

    out_path = Path(args.output)
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()

