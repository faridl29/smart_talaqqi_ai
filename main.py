#!/Users/miftahfaridlal-anshari/.pyenv/versions/3.10.18/bin/python
# ^ Shebang: agar script selalu pakai pyenv Python yang punya torch/transformers.

import os

# Load .env file jika ada (safe fallback)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import signal
# Ctrl+C langsung hard kill (graceful shutdown uvicorn sering macet karena inference model)
signal.signal(signal.SIGINT, lambda s, f: os._exit(0))
signal.signal(signal.SIGTERM, lambda s, f: os._exit(0))

import asyncio
import time
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
    description="Real-Time WebSocket ASR & Makhraj AI Server dengan Tarteel Whisper Tiny",
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
    has_spoken_recently = False

    # Stream Tuning membaca dari .env (dengan fallback aman yang dioptimasi untuk VPS 2 vCPU)
    last_audio_eval_size = 0
    MIN_BUFFER_BYTES = int(os.getenv("MIN_BUFFER_BYTES", "32000"))
    AUDIO_EVAL_INTERVAL_BYTES = int(os.getenv("AUDIO_EVAL_INTERVAL_BYTES", "32000"))
    EVAL_WINDOW_BYTES = 32000         # VAD check 1.0 detik terakhir
    STREAM_WINDOW_BYTES = 1024000     # ~32 detik akumulasi audio (full recitation context)
    VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "450"))
    is_evaluating = False

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
                    has_spoken_recently = False
                    is_evaluating = False
                    is_auto_completed_sent = False
                    session_matched_indices = set()
                    logger.info(f"[WebSocket] Init Talaqqi Stream untuk Target Ayat: '{target_ayah_text}'")
                    await websocket.send_json({
                        "status": "initialized",
                        "tarteel_ready": TarteelASR.is_available(),
                        "message": "Server AI siap menerima stream audio (Tarteel Whisper Tiny)."
                    })

                elif action == "finish":
                    if target_ayah_text:
                        if len(audio_pcm_buffer) > 0 and TarteelASR.is_available():
                            target_words = len(target_ayah_text.split())
                            dynamic_tokens = max(32, min(256, target_words * 6 + 30))
                            # Gunakan SELURUH audio_pcm_buffer (bukan hanya 15s) agar ayat panjang dari awal hingga akhir ter-evaluasi!
                            final_audio = bytes(audio_pcm_buffer)
                            t0 = time.time()
                            raw_transcript, confidence = TarteelASR.transcribe_pcm(
                                final_audio,
                                max_tokens=dynamic_tokens
                            )
                            t_asr = time.time() - t0
                            final_result = MakhrajEngine.evaluate_realtime_stream(
                                target_ayah_text=target_ayah_text,
                                recognized_speech_text=raw_transcript
                            )
                            final_result["raw_transcript"] = raw_transcript
                            final_result["recognized_speech_text"] = raw_transcript
                            final_result["source"] = "tarteel"
                            final_result["status"] = "completed"
                            final_result["model_confidence"] = round(confidence, 3)
                            logger.info(f"[WebSocket Finish] ⏱️ ASR Finish: {t_asr:.3f}s -> Accuracy: {final_result.get('accuracy')}%, Transkrip: '{raw_transcript}'")
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
                if is_auto_completed_sent:
                    continue

                raw_chunk = msg["bytes"]
                audio_pcm_buffer.extend(raw_chunk)

                buffer_size = len(audio_pcm_buffer)
                if (
                    target_ayah_text
                    and not is_evaluating
                    and buffer_size >= MIN_BUFFER_BYTES
                    and (buffer_size - last_audio_eval_size) >= AUDIO_EVAL_INTERVAL_BYTES
                ):
                    is_evaluating = True
                    last_audio_eval_size = buffer_size

                    try:
                        # VAD: Cek suara manusia pada window 1.0 detik terakhir
                        audio_window = bytes(audio_pcm_buffer[-EVAL_WINDOW_BYTES:])
                        is_speech_now = _has_speech(audio_window)

                        if not is_speech_now:
                            if not has_spoken_recently:
                                continue
                            # Evaluasi final trailing-silence begitu pengguna selesai bicara
                            has_spoken_recently = False
                        else:
                            has_spoken_recently = True

                        if TarteelASR.is_available():
                            # Optimasi Kunci Real-Time: Batasi audio window streaming maks 3.0 detik (96,000 bytes)
                            # Ini membuat waktu eksekusi PyTorch SELALU konsisten ~0.15 detik
                            STREAM_AUDIO_WINDOW_BYTES = 96000
                            stream_audio = bytes(audio_pcm_buffer[-STREAM_AUDIO_WINDOW_BYTES:])
                            
                            t0 = time.time()
                            raw_transcript, confidence = TarteelASR.transcribe_pcm(
                                stream_audio, return_confidence=False, max_tokens=24
                            )
                            t_asr = time.time() - t0

                            # Tarteel output Arabic text dengan tashkeel → transkrip bersih per frame
                            clean = (raw_transcript or "").strip()
                            clean = " ".join(clean.split())
                            if not clean or not any('\u0600' <= c <= '\u06FF' for c in clean):
                                continue

                            target_words_list = [w for w in target_ayah_text.strip().split() if w]
                            rec_words_list = [w for w in clean.split() if w]

                            # Match recognized words 1-to-1 to best matching target words
                            used_target_indices = set()
                            for r_w in rec_words_list:
                                best_sim = 0.0
                                best_t_idx = -1
                                for t_idx, t_w in enumerate(target_words_list):
                                    if t_idx in used_target_indices:
                                        continue
                                    sim = MakhrajEngine._word_similarity(t_w, r_w)
                                    if sim > best_sim and sim >= 0.60:
                                        best_sim = sim
                                        best_t_idx = t_idx
                                if best_t_idx != -1:
                                    used_target_indices.add(best_t_idx)
                                    session_matched_indices.add(best_t_idx)

                            target_total = len(target_words_list)
                            matched_words = len(session_matched_indices)

                            eval_result = MakhrajEngine.evaluate_realtime_stream(
                                target_ayah_text=target_ayah_text,
                                recognized_speech_text=clean
                            )
                            eval_result['matched_count'] = matched_words

                            if target_total > 0 and matched_words >= target_total:
                                eval_result["status"] = "auto_completed"
                                is_auto_completed_sent = True

                                # Evaluasi SELURUH audio_pcm_buffer dari awal hingga akhir (ayat panjang)
                                dynamic_tokens = max(32, min(256, target_total * 6 + 30))
                                final_audio = bytes(audio_pcm_buffer)
                                final_raw, final_conf = TarteelASR.transcribe_pcm(final_audio, max_tokens=dynamic_tokens)
                                if final_raw and any('\u0600' <= c <= '\u06FF' for c in final_raw):
                                    final_eval = MakhrajEngine.evaluate_realtime_stream(
                                        target_ayah_text=target_ayah_text,
                                        recognized_speech_text=final_raw
                                    )
                                    final_eval["status"] = "auto_completed"
                                    final_eval["raw_transcript"] = final_raw
                                    final_eval["recognized_speech_text"] = final_raw
                                    final_eval["source"] = "tarteel"
                                    final_eval["model_confidence"] = round(final_conf, 3)
                                    eval_result = final_eval
                            else:
                                eval_result["status"] = "evaluating"
                                eval_result["raw_transcript"] = clean

                            eval_result["source"] = "tarteel"
                            eval_result["progress"] = f"{matched_words}/{target_total}"
                            eval_result["is_realtime"] = True

                            logger.info(
                                f"⏱️ [WebSocket Stream] ASR: {t_asr:.3f}s | Status ({eval_result['status']}): '{eval_result.get('raw_transcript', clean)}' "
                                f"-> Match: {matched_words}/{target_total} ({eval_result.get('accuracy')}%)"
                            )
                            await websocket.send_json(eval_result)
                    finally:
                        is_evaluating = False
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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
