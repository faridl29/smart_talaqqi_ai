FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies & install
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application files
COPY . .

# Set environment variables for ONNX CPU optimization & HuggingFace cache location
ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# Expose FastAPI port
EXPOSE 8000

# Healthcheck — auto-restart jika server down
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Gunicorn + Uvicorn workers (2 workers untuk double concurrent capacity)
# Setiap worker ~800MB-1.2GB RAM, 2 workers aman di 4GB VPS.
# Upgrade ke 8GB jika ingin 3-4 workers.
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "30"]
