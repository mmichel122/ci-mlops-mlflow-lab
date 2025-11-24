# MLOps End-to-End Pipeline

This project demonstrates a complete end-to-end **MLOps workflow** integrating:

- **MLflow** (experiment tracking, model registry)
- **DVC** (data versioning)
- **MinIO** (object storage)
- **FastAPI** (model serving)
- **Docker** (containerization)
- **Kubernetes / k3s** (model deployment)
- **GitHub Actions** (CI/CD)

It provides a production-like, self-contained stack for managing dataset versions, training models, tracking experiments, registering models, containerizing inference servers, deploying in Kubernetes, and updating models through automated workflows.

---

# 🧱 Architecture Overview

```
Local Dev ──────────────→ GitHub Repo
      │                         │
      │                         ▼
  DVC + MLflow           GitHub Actions CI
      │                         │
      ▼                         ▼
   MinIO (Data)   →   Build & Push Docker Images
      │                         │
      │                         ▼
      ▼                 k3s Cluster Deployment
MLflow Tracking    +   FastAPI Model Server
```

---

# 📦 Project Structure

```
├── src/
│   ├── train_classifier.py      # ML training + MLflow logging + best model selection
│   ├── serve.py                 # FastAPI model inference server
│
├── data/
│   └── raw/                     # Dataset (DVC-tracked)
│
├── dvc.yaml                     # DVC pipeline definition
├── .dvc/                        # DVC metadata
│
├── Dockerfile                   # FastAPI app container
├── requirements.txt             # Python dependencies
│
├── k8s/
│   ├── mlflow.yaml              # MLflow server
│   ├── minio.yaml               # MinIO object storage
│   ├── api.yaml                 # FastAPI deployment + service
│   ├── namespace.yaml           # MLOps namespace
│   ├── service.yaml             # API service
│   ├── secrets.yaml             # MinIO + MLflow secrets
│   └── ingress.yaml             # Optional ingress
│
└── README.md
```

---

# 📊 Data Versioning with DVC

### 1. Track data
```
dvc add data/raw/dataset.csv
```

### 2. Configure remote (MinIO)
```
dvc remote add -d minio s3://dvc-remote
```

### 3. Push data to remote
```
dvc push
```

---

# 🔥 Training Pipeline

The training loop:

- Loads data (local or via `dvc pull` if missing)
- Runs **5 training runs** with seeds `[42 → 46]`
- Logs metrics, parameters, confusion matrix
- Selects **best model** by accuracy
- Logs model artifacts uniquely using:

```
model_<run_id>
```
- Registers **only the best model** in MLflow Model Registry
- Adds descriptive metadata + tags

### Run training locally
```
python src/train_classifier.py --register-model
```

### Training inside Kubernetes (from API)
```
POST http://<api>/train
```

---

# 🧠 MLflow Tracking + Registry

Includes:

- experiment auto-versioning
- automatic creation of experiments
- model artifact tagging
- detailed model version description
- automatic latest-version resolution in API

### MLflow UI
Forward port:

```
kubectl port-forward svc/mlflow 5000:5000 -n mlops
```

Then open:

```
http://localhost:5000
```

---

# 🚀 FastAPI Model Serving

The `serve.py` app:

- Loads latest `models:/<name>/<version>` from MLflow
- Offers `/health`, `/predict`, and `/train`
- On `/train`, retrains in-cluster and reloads the newest model

### Run locally
```
uvicorn src.serve:app --reload
```

---

# 🐳 Dockerization

Build locally:
```
docker build -t mlops-api:latest .
```

Push to Docker Hub:
```
docker push mmdocker06/mlops-lab-api:latest
```

---

# ☸ Kubernetes Deployment

### Apply manifests
```
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/service.yaml
```

### Verify API
```
kubectl port-forward svc/mlops-api 8000:8000 -n mlops
```
Then test:
```
curl -X POST http://localhost:8000/predict
```

---

# 🔄 CI/CD with GitHub Actions

Pipeline covers:

- Build + push Docker image",

