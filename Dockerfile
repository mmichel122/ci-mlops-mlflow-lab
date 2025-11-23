# =========================
# Stage 1: builder
# =========================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# =========================
# Stage 2: runtime
# =========================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder image
COPY --from=builder /install /usr/local

# ⬇️ Copy the whole repo (code + dvc.yaml + .dvc + data metadata)
# .dockerignore will prevent big stuff (raw CSVs, cache, etc.) from being copied
COPY . .

# Default envs (override in k8s/compose as needed)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
# ❗ Remove default MODEL_URI – we resolve/learn it dynamically or via /train
# ENV MODEL_URI=models:/dvc_rf_classifier/1

EXPOSE 8000

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
