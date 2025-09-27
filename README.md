# Titanic Survival Prediction

End-to-end ML pipeline to predict passenger survival on the RMS Titanic using the Kaggle dataset (Titanic: Machine Learning from Disaster).

What's inside
- Data cleaning and feature engineering
- EDA with plots (reports/figures)
- Scikit-learn preprocessing pipelines
- Model comparison (LogReg, RF, GB, SVM, KNN)
- Class imbalance handled via class weights
- Hyperparameter tuning and threshold optimization
- Final model selection and reporting

---

## Project Structure

```
data/
  raw/        # Original Kaggle CSVs (train.csv, test.csv)
  processed/  # Cleaned outputs from 01_cleaning.ipynb
notebooks/
  01_cleaning.ipynb    # Parsing, imputations, feature engineering
  02_modeling.ipynb    # Pipelines, CV, weighting, tuning, thresholding
  03_eda.ipynb         # Additional EDA
src/
  features.py          # Column groups + preprocessing pipelines
  models.py            # Model zoo (logreg, rf, gb, svm, knn)
models/
  best_gb_pipeline.joblib   # Saved final pipeline
reports/
  figures/             # All EDA & result plots
  Titanic_Survival_Prediction_Report__6_.pdf  # Final report
README.md
requirements.txt
```

Dataset: https://www.kaggle.com/c/titanic

## Requirements

- Python 3.10-3.12.
- Install dependencies:

```
pip install -r requirements.txt
```

## Quickstart

1) Create and activate a virtual environment

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Run the notebooks in order

- Open Jupyter: `jupyter lab` (or `jupyter notebook`)
- Run `notebooks/01_cleaning.ipynb` -> writes cleaned data to `data/processed/`
- Run `notebooks/02_modeling.ipynb` -> trains models, saves figures to `reports/figures/`, and persists the best pipeline to `models/best_gb_pipeline.joblib`

### CLI (headless) usage

Train and save the model without notebooks:

```
python -m src.train --train-path data/processed/train_clean.csv --model-out models/best_gb_pipeline.joblib
```

Predict on a cleaned CSV:

```
python -m src.predict --input data/processed/test_clean.csv --model models/best_gb_pipeline.joblib --output submission.csv --write-proba
```

## Using the Saved Model

Example: load the trained pipeline and score the processed test data.

```python
import joblib
import pandas as pd

pipe = joblib.load("models/best_gb_pipeline.joblib")
df_test = pd.read_csv("data/processed/test_clean.csv")

# Predict probabilities of survival
proba = pipe.predict_proba(df_test)[:, 1]
pred = (proba >= 0.47).astype(int)  # tuned threshold
print(pred[:10])
```

## Data Cleaning & Feature Engineering (summary)

- Parsed: Title (from Name), Deck (from Cabin), TicketPrefix (from Ticket)
- Engineered: FamilySize, IsAlone, HasCabin, TicketGroupSize
- Imputed: Embarked (mode), Fare (median per Pclass)
- Age left unimputed in data; median imputation is applied inside the pipeline to avoid leakage
- Dropped high-cardinality identifiers: Name, Ticket, Cabin

Final modeling columns: Pclass, Sex, Age, SibSp, Parch, FamilySize, IsAlone, Fare, Embarked, Deck, HasCabin, TicketGroupSize, TicketPrefix, Title.

## Results (validation split ~20%)

- Best model: Gradient Boosting (GB)
- Notable settings: n_estimators ~ 1141, learning_rate = 0.05, subsample = 0.8
- Class weights: {0: 0.81, 1: 1.30}
- Threshold optimization: best around 0.47

Final performance
- Accuracy: 0.816
- Macro-F1: 0.81
- Class 0 (non-survivors): P=0.86, R=0.84, F1=0.85
- Class 1 (survivors):  P=0.75, R=0.78, F1=0.77

## Figures

Selected plots are stored in reports/figures/:
- missing_values__by_column.png
- Survival_by_sex.png
- correlation_heatmap_numeric.png
- top_15_feature_importances_gb.png
- confusion_matrix.png

## Full Report

For detailed methodology, figures, and discussion, see:

reports/Titanic_Survival_Prediction_Report__6_.pdf

## Data

Raw Kaggle CSVs are not stored in the repo. Download from the competition page or via the Kaggle CLI:

```
# install and authenticate kaggle first
kaggle competitions download -c titanic -p data/raw
unzip -o data/raw/titanic.zip -d data/raw
```

## Author

Hasti Aksoy
GitHub: @hasti-aksoy
