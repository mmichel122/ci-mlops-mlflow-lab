# =========================
# Stage 1: builder
# =========================
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed only for building some wheels (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec and install into a local folder (/install)
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

# Copy only necessary app code
COPY src/ ./src/

# Default envs (override in k8s/compose as needed)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_URI=models:/dvc_rf_classifier/1

# Expose FastAPI port
EXPOSE 8000

# Run as non-root (optional but recommended)
RUN useradd -m appuser
USER appuser

# Start FastAPI
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
