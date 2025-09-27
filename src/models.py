# src/models.py
"""Model zoo for Titanic baseline experiments."""

from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_models() -> Dict[str, object]:
    # Light, sensible defaults for quick baselines
    return {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=None),  # lbfgs is default; increased max_iter
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=15),
    }
