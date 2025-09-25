"""
Train a classification model on sklearn's 'wine' dataset and save artifacts.

- Model: RandomForestClassifier (robust with strong baseline performance).
- Saved artifacts: model.joblib and features.json (feature column order used to train).

Execution:
    python train_model.py
"""

# =========================
# Imports
# =========================
import json
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load the 'wine' dataset from sklearn.

    Returns:
        X (np.ndarray): Feature matrix with shape (n_samples, n_features).
        y (np.ndarray): Target vector with shape (n_samples,).
        feature_names (list[str]): List of feature names.
    
    Side effects:
        None.
    """
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    return X, y, feature_names


def train_model(X, y):
    """
    Train a simple pipeline: StandardScaler + RandomForestClassifier.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        pipe (Pipeline): Trained pipeline ready to predict.
    
    Side effects:
        Does not persist to disk; trains in-memory only.
    """
    # Stratified train/test split for quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline with scaling (harmless for RF; useful if switching to LR later)
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    # Basic evaluation
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (test): {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred))

    return pipe


def save_artifacts(model, feature_names, out_dir="."):
    """
    Persist the model and the feature order to disk.

    Args:
        model (Pipeline): Trained model/pipeline.
        feature_names (list[str]): Column order used during training.
        out_dir (str): Output directory for artifacts.

    Side effects:
        Creates/overwrites files:
            - model.joblib
            - features.json
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "model.joblib")
    with open(out / "features.json", "w", encoding="utf-8") as f:
        json.dump({"feature_order": feature_names}, f, ensure_ascii=False, indent=2)

    print(f"Model saved to: {out.resolve()}")


def main():
    """Entry point when executing the script directly."""
    X, y, feature_names = load_data()
    model = train_model(X, y)
    save_artifacts(model, feature_names)


if __name__ == "__main__":
    main()