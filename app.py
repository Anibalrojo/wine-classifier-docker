"""
Flask API to serve a classification model using the wine dataset from sklearn.

Endpoints:
- GET  /          : Welcome/status message.
- POST /predict   : Receives JSON with "instances" and returns predictions and probabilities.

Input format:
{
  "instances": [
    {"alcohol": 13.2, "malic_acid": 1.78, ...},  # dict with ALL features
    {...}
  ]
}

Notes:
- Basic validation: verifies that required keys are present and builds X in the trained order.
- Returns sklearn class ids and class names.

Local execution:
    python app.py
"""

# =========================
# Imports
# =========================
import json
from pathlib import Path
from typing import List, Dict, Any
import joblib
import numpy as np
from flask import Flask, jsonify, request
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine

# =========================
# Load artifacts at startup (once)
# =========================
ARTIFACTS_DIR = Path(".")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    raise FileNotFoundError(
        "Artifacts not found (model.joblib / features.json). "
        "Run first: python train_model.py"
    )

MODEL: Pipeline = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURE_ORDER = json.load(f)["feature_order"]

# To map class id to readable name, reuse sklearn's dataset
WINE = load_wine()
CLASS_NAMES = list(WINE.target_names)

# =========================
# Flask initialization
# =========================
app = Flask(__name__)


def _validate_and_build_matrix(instances):
    """
    Validate input JSON and build the feature matrix X in FEATURE order.

    Args:
        instances (List[Dict[str, Any]]): List of instances where each dict maps feature->value. Values can be int or float.

    Returns:
        X (np.ndarray): Feature matrix with shape (n_instances, n_features).

    Side effects:
        Raises ValueError with a clear message if keys are missing/extra or types are invalid.
    """
    if not isinstance(instances, list) or len(instances) == 0:
        raise ValueError("The 'instances' field must be a non-empty list.")

    X_rows = []
    for idx, row in enumerate(instances):
        if not isinstance(row, dict):
            raise ValueError(f"Instance at position {idx} is not a valid JSON object.")
        # Check required keys are present
        missing = [f for f in FEATURE_ORDER if f not in row]
        if missing:
            raise ValueError(f"Missing features in instance {idx}: {missing}")

        # Take only expected features in the correct order
        values = []
        for f in FEATURE_ORDER:
            v = row[f]
            # Robust conversion to float
            try:
                v = float(v)
            except Exception:
                raise ValueError(f"Value of '{f}' in instance {idx} is not numeric.")
            values.append(v)
        X_rows.append(values)

    return np.array(X_rows, dtype=float)


@app.get("/")
def index():
    """
    Welcome/status endpoint.
    
    Returns:
        JSON response with API status and expected features.
    """
    return jsonify(
        {
            "message": "Classification API (wine) is up",
            "expected_features": FEATURE_ORDER,
            "predict_endpoint": "/predict",
        }
    ), 200


@app.post("/predict")
def predict():
    """
    Prediction endpoint.

    Returns:
        JSON response with predictions, class names, and probabilities.
        
        Example input:
        {
          "instances": [
            {"alcohol": 13.2, "malic_acid": 1.78, ...},
            ...
          ]
        }
        
        Example output:
        {
          "predictions": [2, 0, ...],
          "classes": ["class_2", "class_0", ...],
          "probas": [[p0,p1,p2], ...],
          "class_names": ["class_0","class_1","class_2"]
        }
    """
    try:
        payload = request.get_json(force=True, silent=False)
        if "instances" not in payload:
            return jsonify({"error": "JSON body must include the 'instances' key."}), 400

        X = _validate_and_build_matrix(payload["instances"])
        preds = MODEL.predict(X).tolist()

        # Probabilities (if the classifier supports it)
        if hasattr(MODEL, "predict_proba"):
            probas = MODEL.predict_proba(X).tolist()
        else:
            probas = None

        # Map class ids to readable class names
        classes_str = [CLASS_NAMES[i] for i in preds]

        resp = {
            "predictions": preds,
            "classes": classes_str,
            "probas": probas,
            "class_names": CLASS_NAMES,
        }
        return jsonify(resp), 200

    except ValueError as e:
        # Controlled validation errors
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Unexpected errors (simple log)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    # Run only if executing python app.py directly (not in gunicorn, etc.)
    # host='0.0.0.0' is key inside Docker to expose to the outside
    app.run(host="0.0.0.0", port=5000, debug=False)