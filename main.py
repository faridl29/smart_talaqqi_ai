#!/Users/miftahfaridlal-anshari/.pyenv/versions/3.10.18/bin/python
# ^ Shebang: agar script selalu pakai pyenv Python yang punya torch/transformers.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import signal
# Ctrl+C langsung hard kill (graceful shutdown uvicorn sering macet karena inference model)
signal.signal(signal.SIGINT, lambda s, f: os._exit(0))
signal.signal(signal.SIGTERM, lambda s, f: os._exit(0))

import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from makhraj_engine import MakhrajEngine
from whisper_quran_asr import TarteelASR

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TalaqqiAIServer")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ASR models secara aman saat startup server."""
    logger.info("Server starting up — pre-loading Tarteel Whisper Tiny...")
    TarteelASR.preload_model()
    yield

app = FastAPI(
    title="Smart Talaqqi AI Server",
    description="Real-Time WebSocket ASR & Makhraj AI Server dengan IqraEval Wav2Vec2-300M",
    version="2.2.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecitationRequest(BaseModel):
    target_ayah_text: str
    recognized_speech_text: str



@app.get("/")
async def health_check():
    qw_ready = TarteelASR.is_available()
    return {
        "server": "Smart Talaqqi AI Server (Tarteel Whisper Tiny)",
        "status": "healthy" if qw_ready else "loading",
        "primary_asr": "tarteel-ai/whisper-tiny-ar-quran (PyTorch CPU)",
        "tarteel_ready": qw_ready,
        "features": [
            "Quran-Specialized ASR (tarteel-ai/whisper-tiny-ar-quran, Apache-2.0)",
            "Real-Time WebSocket Audio Streaming",
            "Makhraj & Phonetic Error Diagnosis",
            "Harakat & Vokal (a/i/u) Mismatch Check",
            "Mad Thabi'i & Kadar Harakat Duration Check"
        ]
    }


@app.post("/api/v1/talaqqi/evaluate")
async def evaluate_recitation(req: RecitationRequest):
    """
    HTTP REST Endpoint untuk evaluasi bacaan & diagnosa makhraj.
    """
    try:
        logger.info(f"[HTTP REST] Request Evaluasi -> Target: '{req.target_ayah_text[:30]}...' | Recognized: '{req.recognized_speech_text}'")
        result = MakhrajEngine.evaluate_realtime_stream(
            target_ayah_text=req.target_ayah_text,
            recognized_speech_text=req.recognized_speech_text
        )
        logger.info(f"[HTTP REST] Hasil Evaluasi -> Accuracy: {result.get('accuracy')}%, Matched: {result.get('matched_count')}/{result.get('total_words')}, Makhraj Errors: {len(result.get('makhraj_errors', []))}")
        return result
    except Exception as e:
        logger.error(f"Error evaluating recitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/talaqqi/stream")
async def websocket_talaqqi_stream(websocket: WebSocket):
    """
    WebSocket Real-Time Audio Stream Endpoint (Tarteel Whisper Tiny).
    """
    await websocket.accept()
    logger.info("Client WebSocket terkoneksi ke Talaqqi AI Server.")

    target_ayah_text = ""
    audio_pcm_buffer = bytearray()

    # Throttle: evaluasi setiap ~1 detik audio (16kHz 16-bit = 32000 bytes/sec)
    last_audio_eval_size = 0
    MIN_BUFFER_BYTES = 16000          # mulai evaluasi setelah ~0.5 detik
    AUDIO_EVAL_INTERVAL_BYTES = 16000 # interval antar eval ~0.5 detik
    EVAL_WINDOW_BYTES = 48000         # window 1.5 detik terakhir
    VAD_RMS_THRESHOLD = 500           # int16 RMS minimum (naik dari 200: laptop mic punya ambient ~300-400)

    def _has_speech(pcm: bytes) -> bool:
        if not pcm:
            return False
        # RMS sederhana dari int16 samples
        import array
        samples = array.array('h', pcm)
        if not samples:
            return False
        sq = sum(s * s for s in samples)
        return (sq / len(samples)) ** 0.5 > VAD_RMS_THRESHOLD

    try:
        while True:
            msg = await websocket.receive()

            # ─── Handle JSON Control Messages (dipakai saat flutter kirim text frame) ───
            if "text" in msg and msg["text"]:
                data: Dict[str, Any] = json.loads(msg["text"])
                action = data.get("action", "")

                if action == "init":
                    target_ayah_text = data.get("target_ayah_text", "")
                    audio_pcm_buffer.clear()
                    last_audio_eval_size = 0
                    logger.info(f"[WebSocket] Init Talaqqi Stream untuk Target Ayat: '{target_ayah_text}'")
                    await websocket.send_json({
                        "status": "initialized",
                        "tarteel_ready": TarteelASR.is_available(),
                        "message": "Server AI siap menerima stream audio (Tarteel Whisper Tiny)."
                    })

                elif action == "finish":
                    if target_ayah_text:
                        if len(audio_pcm_buffer) > 0 and TarteelASR.is_available():
                            raw_transcript, confidence = TarteelASR.transcribe_pcm(bytes(audio_pcm_buffer))
                            final_result = MakhrajEngine.evaluate_realtime_stream(
                                target_ayah_text=target_ayah_text,
                                recognized_speech_text=raw_transcript
                            )
                            final_result["source"] = "tarteel"
                            final_result["status"] = "completed"
                            final_result["model_confidence"] = round(confidence, 3)
                            logger.info(f"[WebSocket Finish] Evaluasi Selesai -> Accuracy: {final_result.get('accuracy')}%, Transkrip: '{raw_transcript}', Makhraj Errors: {len(final_result.get('makhraj_errors', []))}, Model Conf: {confidence:.2%}")
                            await websocket.send_json(final_result)
                        else:
                            await websocket.send_json({
                                "status": "error",
                                "message": "Tidak ada buffer audio yang diterima oleh Server AI."
                            })
                    break

                continue

            # ─── Handle Binary Audio Stream (Raw PCM 16kHz 16-bit Mono) ───
            if "bytes" in msg and msg["bytes"]:
                raw_chunk = msg["bytes"]
                audio_pcm_buffer.extend(raw_chunk)

                buffer_size = len(audio_pcm_buffer)
                if (
                    target_ayah_text
                    and buffer_size >= MIN_BUFFER_BYTES
                    and (buffer_size - last_audio_eval_size) >= AUDIO_EVAL_INTERVAL_BYTES
                ):
                    last_audio_eval_size = buffer_size

                    # VAD: skip chunk sunyi sebelum inference mahal
                    audio_window = bytes(audio_pcm_buffer[-EVAL_WINDOW_BYTES:])
                    if not _has_speech(audio_window):
                        continue

                    if TarteelASR.is_available():
                        # Transkripsikan seluruh akumulasi audio yang telah dibaca sejauh ini (hingga max 30s = 960000 bytes)
                        accumulated_audio = bytes(audio_pcm_buffer[-960000:])
                        raw_transcript, confidence = TarteelASR.transcribe_pcm(
                            accumulated_audio, return_confidence=False
                        )
                        # Tarteel output Arabic text dengan tashkeel → langsung dipakai.
                        # Skip kalau empty / cuma punctuation.
                        clean = (raw_transcript or "").strip()
                        clean = " ".join(clean.split())
                        if not clean or not any('\u0600' <= c <= '\u06FF' for c in clean):
                            continue

                        eval_result = MakhrajEngine.evaluate_realtime_stream(
                            target_ayah_text=target_ayah_text,
                            recognized_speech_text=clean
                        )
                        eval_result["status"] = "evaluating"
                        eval_result["source"] = "tarteel"
                        eval_result["model_confidence"] = round(confidence, 3)
                        eval_result["raw_transcript"] = clean

                        target_total = eval_result.get('total_words', 0)
                        matched_words = eval_result.get('matched_count', 0)
                        eval_result["progress"] = f"{matched_words}/{target_total}"
                        eval_result["is_realtime"] = True

                        logger.info(
                            f"[WebSocket Stream] Tarteel: '{raw_transcript}' "
                            f"-> Match Words: {matched_words}/{target_total} "
                            f"| Accuracy: {eval_result.get('accuracy')}% "
                            f"| Progress: {eval_result['progress']} | Conf: {confidence:.2%}"
                        )
                        await websocket.send_json(eval_result)
                continue


    except WebSocketDisconnect:
        logger.info("Client WebSocket terputus.")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
