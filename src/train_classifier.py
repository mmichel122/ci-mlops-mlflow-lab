#!/usr/bin/env python
"""
Train a classifier, track experiments with MLflow, data with DVC.

Steps:
- Load dataset from data/raw/dataset.csv
- Run 5 experiments (different random_state seeds)
- Log params, metrics, and confusion matrix plot to MLflow
- Register ONLY the best model in MLflow Model Registry (optional)
"""

import argparse
import subprocess
import os
from pathlib import Path
import re

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

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
        help="Base random seed (we'll derive 5 seeds from this)",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="If set, register the BEST model in the MLflow Model Registry",
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default="dvc_rf_classifier",
        help="Name of the registered model (if --register-model is set)",
    )

    return parser.parse_args()


def _next_experiment_version(name: str) -> str:
    """
    Given a name like 'dvc_mlflow_classifier' or 'dvc_mlflow_classifier_v3',
    return the next versioned name:

      'dvc_mlflow_classifier'    -> 'dvc_mlflow_classifier_v1'
      'dvc_mlflow_classifier_v3' -> 'dvc_mlflow_classifier_v4'
    """
    m = re.match(r"^(.*)_v(\d+)$", name)
    if m:
        base = m.group(1)
        version = int(m.group(2))
        return f"{base}_v{version + 1}"
    else:
        return f"{name}_v1"


def ensure_experiment(experiment_name: str) -> str:
    """
    Try to set the given experiment name. If it fails because the experiment
    is soft-deleted, increment a _vN suffix and retry.

    Returns the final experiment name that was successfully set.
    """
    current = experiment_name
    max_attempts = 10

    for _ in range(max_attempts):
        try:
            mlflow.set_experiment(current)
            if current != experiment_name:
                print(
                    f"[MLFLOW] Original experiment '{experiment_name}' was deleted; "
                    f"using '{current}' instead."
                )
            return current
        except MlflowException as e:
            msg = str(e)
            if "Cannot set a deleted experiment" in msg:
                # Compute next versioned experiment name
                next_name = _next_experiment_version(current)
                print(
                    f"[MLFLOW] Cannot use deleted experiment '{current}'. "
                    f"Trying new experiment name '{next_name}'."
                )
                current = next_name
                continue
            # Some other MlflowException → re-raise
            raise

    raise MlflowException(
        f"Failed to create or set experiment after {max_attempts} attempts, "
        f"starting from '{experiment_name}'."
    )


def main():
    args = parse_args()

    # Point to MLflow server
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # Ensure we have a usable experiment, even if the original was soft-deleted
    final_experiment_name = ensure_experiment(args.experiment_name)

    # Load dataset tracked by DVC
    data_path = Path("data/raw/dataset.csv")

    if not data_path.exists():
        print(f"[DATA] {data_path} not found. Trying plain 'dvc pull'...")
        try:
            # No path argument → pull all tracked outs based on dvc.yaml + *.dvc
            subprocess.run(
                ["dvc", "pull"],
                check=True,
            )
        except Exception as e:
            print(f"[DATA] dvc pull failed: {e!r}")

        # Re-check after dvc pull
        if not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} not found even after 'dvc pull'. "
                "Did you configure your DVC remote and credentials, "
                "and is the dataset tracked in this repo?"
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

    # ------------------------------------------------------------------
    # Run 5 experiments: same hyperparams, different random_state seeds
    # ------------------------------------------------------------------
    seeds = [args.random_state + i for i in range(5)]

    best_acc = -1.0
    best_run_id = None

    print(
        f"[TRAIN] Running 5 experiments in experiment '{final_experiment_name}' "
        f"with seeds: {seeds}"
    )

    for seed in seeds:
        run_name = f"rf_classifier_dvc_seed_{seed}"
        print(f"[TRAIN] Starting run '{run_name}' (random_state={seed})")

        with mlflow.start_run(run_name=run_name) as run:
            # Log hyperparams
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", seed)
            mlflow.log_param("experiment_name", final_experiment_name)

            # Train model
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=seed,
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
            cm_path = os.path.join(
                "artifacts_local", f"confusion_matrix_seed_{seed}.png"
            )

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

            # Log model as an artifact (no registry yet)
            artifact_name = f"model_{run.info.run_id}"

            # Make Run UI show the artifact path
            mlflow.log_param("model_artifact_path", artifact_name)

            mlflow.sklearn.log_model(
                sk_model=model,
                input_example=input_example,
                signature=signature,
                name=artifact_name,
            )

            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            tracking_uri = mlflow.get_tracking_uri()

            print(f"[RUN] Completed run {run_id} with accuracy={acc:.4f}")
            print(
                f"🏃 View run {run_name} at: "
                f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
            )

            # Track best run
            if acc > best_acc:
                best_acc = acc
                best_run_id = run_id
                best_seed = seed

    # ------------------------------------------------------------------
    # After 5 runs: register ONLY the best model (if requested)
    # ------------------------------------------------------------------
    if args.register_model:
        if best_run_id is None:
            raise RuntimeError("No successful runs to register a model from.")

        client = MlflowClient()
        source_uri = f"runs:/{best_run_id}/model_{best_run_id}"

        print(
            f"[REGISTER] Registering best model from run {best_run_id} "
            f"(accuracy={best_acc:.4f}) to '{args.registered_model_name}'"
        )

        registered = mlflow.register_model(
            model_uri=source_uri,
            name=args.registered_model_name,
        )

        # Add useful tags to the model version
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="accuracy",
            value=str(best_acc),
        )

        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="best_run_id",
            value=best_run_id,
        )

        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="artifact_path",
            value=f"model_{best_run_id}",
        )

        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="accuracy",
            value=str(best_acc),
        )

        client.update_model_version(
            name=args.registered_model_name,
            version=registered.version,
            description=(
                f"Model selected from 5 random seeds.\n"
                f"Accuracy: {best_acc:.4f}\n"
                f"Run ID: {best_run_id}\n"
                f"Artifact path: model_{best_run_id}"
            )
        )

        print(
            f"✅ Registered model '{args.registered_model_name}' "
            f"version {registered.version} from run {best_run_id} "
            f"(accuracy={best_acc:.4f})"
        )
    else:
        print(
            f"[RESULT] Best run accuracy={best_acc:.4f} "
            f"(run_id={best_run_id}), no model registered "
            f"(--register-model not set)."
        )


if __name__ == "__main__":
    main()
