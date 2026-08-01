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
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download IqraEval 300M model during build
RUN python -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor; Wav2Vec2CTCTokenizer.from_pretrained('FatimahEmadEldin/wav2vec2-xls-r-300m-iqraeval'); Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-300m'); Wav2Vec2ForCTC.from_pretrained('FatimahEmadEldin/wav2vec2-xls-r-300m-iqraeval')" || true


# Copy application files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run Uvicorn server (1 worker agar Whisper model cukup RAM di 4GB VPS)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
