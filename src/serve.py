#!/usr/bin/env python
"""
FastAPI model server loading a model from MLflow.

Behaviour:

- On startup:
  * Set MLFLOW_TRACKING_URI
  * If MODEL_URI is set -> try to load that model
  * Else -> try to load latest registered version of MODEL_NAME
  * If nothing exists or loading fails -> model stays None

- POST /train:
  * Runs train_classifier.py --register-model
  * Looks up the latest registered version of MODEL_NAME
  * Loads it into memory
  * Returns the resolved model_uri

- POST /predict:
  * Requires that a model is already loaded, otherwise returns 503
"""

import os
import sys
import subprocess
from typing import List, Optional

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np


# --------- Config from environment ---------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Optional explicit URI, e.g. "models:/dvc_rf_classifier/1"
MODEL_URI: Optional[str] = os.getenv("MODEL_URI")

# Name of the registered model in MLflow (must match --registered-model-name in train_classifier.py)
MODEL_NAME = os.getenv("MODEL_NAME", "dvc_rf_classifier")

# Path to training script (what you run manually: python src/train_classifier.py --register-model)
TRAIN_SCRIPT_PATH = os.getenv("TRAIN_SCRIPT_PATH", "src/train_classifier.py")
# -------------------------------------------


class Instances(BaseModel):
    """Input schema: batch of feature vectors."""
    inputs: List[List[float]]


app = FastAPI(title="DVC + MLflow RF Classifier")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Global model object loaded at startup or after /train
model = None


# ---------- Helper functions ----------

def get_latest_registered_model_uri(model_name: str) -> Optional[str]:
    """
    Return a models:/ URI for the latest registered version of `model_name`,
    or None if no versions exist.
    """
    print(f"[MODEL RESOLVE] Looking up latest registered version of '{model_name}'...")
    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
    except MlflowException as e:
        print(f"[MODEL RESOLVE] search_model_versions failed: {e!r}")
        return None

    if not versions:
        print(f"[MODEL RESOLVE] No versions found for '{model_name}'.")
        return None

    latest = max(versions, key=lambda v: int(v.version))
    uri = f"models:/{model_name}/{latest.version}"
    print(f"[MODEL RESOLVE] Found registered model URI: {uri}")
    return uri


def run_training_script_once() -> None:
    """
    Runs the existing training script to create the experiment + registered model.

    Equivalent to:
        python src/train_classifier.py --register-model
    """
    cwd = os.getcwd()
    cmd = [sys.executable, TRAIN_SCRIPT_PATH, "--register-model"]

    print(f"[TRAIN] Running training script in cwd={cwd}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print("[TRAIN] stdout:\n", result.stdout)
    print("[TRAIN] stderr:\n", result.stderr)

    if result.returncode != 0:
        raise MlflowException(
            f"Training script failed with exit code {result.returncode}"
        )


def try_load_model_once() -> Optional[str]:
    """
    Try to resolve a model URI and load it.

    - If MODEL_URI env is set -> try that.
    - Else -> try latest registered version of MODEL_NAME.

    Returns the URI used if successful, otherwise None (model stays unloaded).
    """
    global MODEL_URI, model

    # 1) Explicit MODEL_URI from env
    if MODEL_URI:
        print(f"[LOAD] Trying explicit MODEL_URI: {MODEL_URI}")
        try:
            loaded = mlflow.pyfunc.load_model(MODEL_URI)
            model = loaded
            print("[LOAD] Model loaded successfully from explicit MODEL_URI.")
            return MODEL_URI
        except Exception as e:
            print(f"[LOAD] Failed to load MODEL_URI {MODEL_URI!r}: {e!r}")
            model = None
            return None

    # 2) Try latest registered version
    uri = get_latest_registered_model_uri(MODEL_NAME)
    if not uri:
        print("[LOAD] No registered model found to load.")
        model = None
        return None

    try:
        print(f"[LOAD] Loading model from resolved URI: {uri}")
        loaded = mlflow.pyfunc.load_model(uri)
        model = loaded
        MODEL_URI = uri
        print("[LOAD] Model loaded successfully from registry.")
        return uri
    except Exception as e:
        print(f"[LOAD] Failed to load model from URI {uri!r}: {e!r}")
        model = None
        return None


# ---------- FastAPI lifecycle + endpoints ----------

@app.on_event("startup")
def startup_load_model() -> None:
    """
    On application startup:
      - Attempt to load an existing model (no bootstrap).
      - If nothing exists or loading fails, model stays None.
    """
    print(f"[STARTUP] MLFLOW_TRACKING_URI = {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    uri = try_load_model_once()
    if uri is None:
        print(
            "[STARTUP] No model loaded. Use POST /train to run training and "
            "register a model before calling /predict."
        )


@app.get("/health")
def health():
    """
    Health endpoint:
      - status: "ok" if model is loaded, otherwise "model_missing"
      - tracking_uri: which MLflow server we’re pointing to
      - model_uri: the resolved model URI (if any)
    """
    return {
        "status": "ok" if model is not None else "model_missing",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MODEL_URI,
    }


@app.post("/train")
def train():
    """
    Trigger training via train_classifier.py and reload the latest model.

    1. Runs: python TRAIN_SCRIPT_PATH --register-model
    2. Looks up latest registered version of MODEL_NAME
    3. Loads it into memory
    """
    global model, MODEL_URI

    try:
        run_training_script_once()
        uri = get_latest_registered_model_uri(MODEL_NAME)
        if uri is None:
            raise MlflowException(
                "Training completed but no registered model was found."
            )

        print(f"[TRAIN] Reloading model from newly registered URI: {uri}")
        loaded = mlflow.pyfunc.load_model(uri)
        model = loaded
        MODEL_URI = uri

        return {
            "status": "trained",
            "model_uri": uri,
        }

    except Exception as e:
        print("[TRAIN] Training or reload failed:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {e}",
        )


@app.post("/predict")
def predict(instances: Instances):
    """
    Predict endpoint.

    Expects:
      {
        "inputs": [
          [feature1, feature2, ...],
          [feature1, feature2, ...]
        ]
      }
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Call POST /train first.",
        )

    X = np.array(instances.inputs)
    try:
        preds = model.predict(X)
    except Exception as e:
        print("Prediction error:", repr(e))
        raise HTTPException(status_code=400, detail=str(e))

    return {"predictions": [int(p) for p in preds]}
