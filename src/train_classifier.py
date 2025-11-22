#!/usr/bin/env python
"""
Train a classifier, track experiments with MLflow, data with DVC.

Steps:
- Load dataset from data/raw/dataset.csv
- Train a simple classifier (RandomForestClassifier)
- Log params, metrics, and confusion matrix plot to MLflow
- Register best model in MLflow Model Registry (optional)
"""

import argparse
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="MLflow + DVC classifier training")

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="dvc_mlflow_classifier",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in RandomForest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max depth of trees",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="If set, register this model in the MLflow Model Registry",
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default="dvc_rf_classifier",
        help="Name of the registered model (if --register-model is set)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Point to MLflow server
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment_name)

    # Load dataset tracked by DVC
    data_path = "data/raw/dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Did you run `dvc pull` and have dataset.csv?"
        )

    df = pd.read_csv(data_path)

    # Assume last column is target, rest are features
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    with mlflow.start_run(run_name="rf_classifier_dvc") as run:
        # Log hyperparams
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        # Plot confusion matrix
        os.makedirs("artifacts_local", exist_ok=True)
        cm_path = os.path.join("artifacts_local", "confusion_matrix.png")

        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        # Log confusion matrix as artifact
        mlflow.log_artifact(cm_path, artifact_path="plots")

                # ---- Model signature + input_example ----
        input_example = X_test[:5]
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model in modern style:
        # - name="model" replaces artifact_path="model"
        # - registered_model_name auto-registers if requested
        logged = mlflow.sklearn.log_model(
            sk_model=model,
            input_example=input_example,
            signature=signature,
            name="model",
            registered_model_name=(
                args.registered_model_name if args.register_model else None
            ),
        )

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        tracking_uri = mlflow.get_tracking_uri()

        print(f"Run completed. accuracy = {acc:.4f}")
        print("Check the MLflow UI for details.")
        print(
            f"🏃 View run rf_classifier_dvc at: "
            f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
        )
        print(f"📦 Model URI: {logged.model_uri}")

        if args.register_model:
            print(
                f"✅ Registered model '{args.registered_model_name}' "
                f"from run {run_id} (accuracy={acc:.4f})"
            )



if __name__ == "__main__":
    main()
