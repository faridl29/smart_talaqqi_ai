FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# ffmpeg: required by faster-whisper for audio decoding
# libsndfile1: required by soundfile package
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run Uvicorn server (1 worker agar Whisper model cukup RAM di 4GB VPS)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
