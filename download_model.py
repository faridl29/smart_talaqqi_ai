import os
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILENAME = "wav2vec2_quran_quantized.onnx"
VOCAB_FILENAME = "vocab.json"

# Repository HuggingFace model Wav2Vec2 Quran ONNX
REPO_ID = "onnx-community/whisper-tiny-ar" # or "tarteel/wav2vec2-arabic-quran"

def download_onnx_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    vocab_path = os.path.join(MODEL_DIR, VOCAB_FILENAME)

    if os.path.exists(model_path):
        logger.info(f"Model ONNX sudah ada di: {model_path}")
        return model_path

    logger.info(f"Mengunduh file model ONNX dari HuggingFace repository: {REPO_ID}...")
    try:
        # Download model ONNX ringan (< 150 MB)
        downloaded_model = hf_hub_download(
            repo_id="Onyx-AI/wav2vec2-large-xlsr-53-quran-onnx",
            filename="model_quantized.onnx",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        os.rename(downloaded_model, model_path)
        logger.info(f"Model ONNX berhasil diunduh ke: {model_path}")
        return model_path
    except Exception as e:
        logger.warning(f"Tidak dapat mengunduh model dari HuggingFace ({e}). Menggunakan Engine Fonetis Bawaan.")
        return None

if __name__ == "__main__":
    download_onnx_model()
