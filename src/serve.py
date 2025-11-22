#!/usr/bin/env python
"""
Simple FastAPI model server loading a model from MLflow.

- Loads model from a given MLflow URI (e.g. models:/dvc_rf_classifier/1)
- Exposes /predict endpoint for inference
"""

import os
from typing import List

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

# --------- Config ---------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/dvc_rf_classifier/1")
# --------------------------


class Instances(BaseModel):
    inputs: List[List[float]]  # 2D list: batch of feature vectors


app = FastAPI(title="DVC + MLflow RF Classifier")

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Loading model from {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(instances: Instances):
    # Convert input to numpy array
    X = np.array(instances.inputs)
    preds = model.predict(X)
    # Convert numpy types to Python types
    return {"predictions": [int(p) for p in preds]}
