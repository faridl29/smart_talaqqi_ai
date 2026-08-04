#!/Users/miftahfaridlal-anshari/.pyenv/versions/3.10.18/bin/python
"""
Smart Talaqqi AI Server — Main FastAPI & WebSocket Entry Point.
Production-grade Real-Time Speech Recognition & Makhraj Diagnostics.
"""

import os
import signal
import asyncio
import array
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Set

# Environment & Math thread limits for 2 vCPU VPS deployment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Signal handling for instantaneous terminate signal
signal.signal(signal.SIGINT, lambda s, f: os._exit(0))
signal.signal(signal.SIGTERM, lambda s, f: os._exit(0))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from makhraj_engine import MakhrajEngine
from whisper_quran_asr import TarteelASR

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TalaqqiAIServer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ASR models safely during server startup."""
    logger.info("Server starting up — pre-loading Tarteel Quran ASR Native ONNX...")
    TarteelASR.preload_model()
    yield


app = FastAPI(
    title="Smart Talaqqi AI Server",
    description="Real-Time WebSocket ASR & Makhraj AI Server dengan Tarteel Quran ONNX Runtime",
    version="2.3.0",
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
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint indicating detailed server state and loaded engine."""
    qw_ready = TarteelASR.is_available()
    return {
        "server": "Smart Talaqqi AI Server (Tarteel Quran ASR)",
        "status": "healthy" if qw_ready else "loading",
        "primary_asr": f"{TarteelASR.MODEL_ID} (Pure Native ONNX CPU)",
        "tarteel_ready": qw_ready,
        "smart_talaqqi_ready": qw_ready,
        "features": [
            "Quran-Specialized ASR (tarteel-ai/whisper-base-ar-quran, Apache-2.0)",
            "Native ONNX Runtime Zero-Torch Pipeline",
            "Real-Time WebSocket Audio Streaming with Adaptive VAD",
            "Makhraj & Phonetic Error Diagnosis",
            "Harakat & Vokal (a/i/u) Mismatch Check",
            "Mad Thabi'i & Kadar Harakat Duration Check"
        ]
    }


@app.post("/api/v1/talaqqi/evaluate")
async def evaluate_recitation(req: RecitationRequest) -> Dict[str, Any]:
    """
    HTTP REST Endpoint for offline/post-recitation evaluation & makhraj diagnosis.
    """
    try:
        logger.info(
            f"[HTTP REST] Request Evaluasi -> Target: '{req.target_ayah_text[:30]}...' | "
            f"Recognized: '{req.recognized_speech_text}'"
        )
        result = MakhrajEngine.evaluate_realtime_stream(
            target_ayah_text=req.target_ayah_text,
            recognized_speech_text=req.recognized_speech_text
        )
        logger.info(
            f"[HTTP REST] Hasil Evaluasi -> Accuracy: {result.get('accuracy')}%, "
            f"Matched: {result.get('matched_count')}/{result.get('total_words')}, "
            f"Makhraj Errors: {len(result.get('makhraj_errors', []))}"
        )
        return result
    except Exception as e:
        logger.error(f"Error evaluating recitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/talaqqi/stream")
async def websocket_talaqqi_stream(websocket: WebSocket) -> None:
    """
    WebSocket Real-Time Audio Stream Endpoint (Tarteel Quran ASR).
    Receives raw PCM 16kHz 16-bit mono stream and returns real-time makhraj evaluation.
    """
    await websocket.accept()
    logger.info("Client WebSocket terkoneksi ke Talaqqi AI Server.")

    target_ayah_text: str = ""
    audio_pcm_buffer: bytearray = bytearray()
    has_spoken_recently: bool = False

    # Stream Tuning configuration (VPS 2 vCPU CPU optimization)
    last_audio_eval_size: int = 0
    MIN_BUFFER_BYTES: int = int(os.getenv("MIN_BUFFER_BYTES", "32000"))
    AUDIO_EVAL_INTERVAL_BYTES: int = int(os.getenv("AUDIO_EVAL_INTERVAL_BYTES", "32000"))
    EVAL_WINDOW_BYTES: int = 32000         # VAD check 1.0s window
    VAD_RMS_THRESHOLD: int = int(os.getenv("VAD_RMS_THRESHOLD", "450"))
    is_evaluating: bool = False
    is_auto_completed_sent: bool = False

    # Session state tracking for Flutter client sync
    session_matched_indices: Set[int] = set()
    session_transcribed_words: Dict[int, str] = {}
    session_word_similarities: Dict[int, float] = {}

    # Adaptive noise baseline calibration
    noise_baseline_rms: float = 0.0
    noise_calibrated: bool = False

    def _has_speech(pcm: bytes) -> bool:
        nonlocal noise_baseline_rms, noise_calibrated
        if not pcm:
            return False
        samples = array.array('h', pcm)
        if not samples:
            return False

        # RMS Energy
        sq = sum(s * s for s in samples)
        rms = (sq / len(samples)) ** 0.5

        # Noise baseline calibration from initial audio
        if not noise_calibrated and len(audio_pcm_buffer) <= EVAL_WINDOW_BYTES * 2:
            noise_baseline_rms = max(noise_baseline_rms, rms * 0.8)
            noise_calibrated = len(audio_pcm_buffer) >= EVAL_WINDOW_BYTES

        adaptive_threshold = max(VAD_RMS_THRESHOLD, noise_baseline_rms * 2.0)
        if rms <= adaptive_threshold:
            return False

        # Zero-Crossing Rate check
        zero_crossings = sum(1 for i in range(1, len(samples)) if (samples[i] >= 0) != (samples[i - 1] >= 0))
        zcr = zero_crossings / len(samples)
        if zcr > 0.35:
            return False  # Likely noise rather than speech

        return True

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.info("[WebSocket] Timeout 30s tanpa data, menutup koneksi.")
                break

            # ─── JSON Control Frame Handling ───
            if "text" in msg and msg["text"]:
                data: Dict[str, Any] = json.loads(msg["text"])
                action: str = data.get("action", "")

                if action == "ping":
                    continue

                if action == "init":
                    target_ayah_text = data.get("target_ayah_text", "")
                    audio_pcm_buffer.clear()
                    last_audio_eval_size = 0
                    has_spoken_recently = False
                    is_evaluating = False
                    is_auto_completed_sent = False
                    session_matched_indices.clear()
                    session_transcribed_words.clear()
                    session_word_similarities.clear()
                    logger.info(f"[WebSocket] Init Talaqqi Stream untuk Target Ayat: '{target_ayah_text}'")
                    await websocket.send_json({
                        "status": "initialized",
                        "tarteel_ready": TarteelASR.is_available(),
                        "message": "Server AI siap menerima stream audio (Tarteel Quran Native ONNX)."
                    })

                elif action == "finish":
                    if target_ayah_text:
                        if len(audio_pcm_buffer) > 0 and TarteelASR.is_available():
                            target_words = len(target_ayah_text.split())
                            dynamic_tokens = max(32, min(256, target_words * 6 + 30))
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
                            logger.info(
                                f"[WebSocket Finish] ⏱️ ASR Finish: {t_asr:.3f}s -> "
                                f"Accuracy: {final_result.get('accuracy')}%, Transkrip: '{raw_transcript}'"
                            )
                            await websocket.send_json(final_result)
                        else:
                            await websocket.send_json({
                                "status": "error",
                                "message": "Tidak ada buffer audio yang diterima oleh Server AI."
                            })
                    break

                continue

            # ─── Binary Audio Stream Handling (Raw PCM 16kHz 16-bit Mono) ───
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
                        # VAD on last 1.0s window
                        audio_window = bytes(audio_pcm_buffer[-EVAL_WINDOW_BYTES:])
                        is_speech_now = _has_speech(audio_window)

                        if not is_speech_now:
                            if not has_spoken_recently:
                                continue
                            has_spoken_recently = False
                        else:
                            has_spoken_recently = True

                        if TarteelASR.is_available():
                            # Adaptive audio window: 4s to 6s for intermediate streaming, reducing latency by ~50%
                            STREAM_AUDIO_WINDOW_BYTES = min(len(audio_pcm_buffer), 80000)
                            stream_audio = bytes(audio_pcm_buffer[-STREAM_AUDIO_WINDOW_BYTES:])

                            t0 = time.time()
                            raw_transcript, confidence = TarteelASR.transcribe_pcm(
                                stream_audio, return_confidence=False, max_tokens=24
                            )
                            t_asr = time.time() - t0

                            clean = (raw_transcript or "").strip()
                            clean = " ".join(clean.split())
                            if not clean or not any('\u0600' <= c <= '\u06FF' for c in clean):
                                continue

                            target_words_list = [w for w in target_ayah_text.strip().split() if w]
                            rec_words_list = [w for w in clean.split() if w]

                            used_target_indices: Set[int] = set()
                            last_matched_idx = max(session_matched_indices) if session_matched_indices else 0

                            for r_w in rec_words_list:
                                best_sim = 0.0
                                best_t_idx = -1
                                for t_idx, t_w in enumerate(target_words_list):
                                    if t_idx in used_target_indices:
                                        continue
                                    if t_idx > last_matched_idx + 2:
                                        continue
                                    sim = MakhrajEngine._word_similarity(t_w, r_w)
                                    if sim > best_sim and sim >= 0.65:
                                        best_sim = sim
                                        best_t_idx = t_idx
                                if best_t_idx != -1:
                                    used_target_indices.add(best_t_idx)
                                    session_matched_indices.add(best_t_idx)
                                    old_sim = session_word_similarities.get(best_t_idx, 0.0)
                                    if best_sim >= old_sim:
                                        session_transcribed_words[best_t_idx] = r_w
                                        session_word_similarities[best_t_idx] = best_sim
                                    last_matched_idx = max(last_matched_idx, best_t_idx)

                            target_total = len(target_words_list)
                            matched_words = len(session_matched_indices)

                            ordered_words = [session_transcribed_words[i] for i in sorted(session_transcribed_words.keys())]
                            accumulated_text = " ".join(ordered_words)

                            if len(rec_words_list) < target_total and matched_words >= target_total:
                                matched_words = target_total - 1

                            eval_result = MakhrajEngine.evaluate_realtime_stream(
                                target_ayah_text=target_ayah_text,
                                recognized_speech_text=accumulated_text if len(accumulated_text.split()) > len(clean.split()) else clean
                            )

                            if len(rec_words_list) < target_total and matched_words >= target_total:
                                matched_words = target_total - 1
                            
                            actual_matched = sum(
                                 1 for w in eval_result.get("word_results", [])
                                 if isinstance(w, dict) and w.get("status") == "matched"
                             )
                            eval_result["matched_count"] = actual_matched
                            eval_result["source"] = "tarteel"
                            eval_result["progress"] = f"{actual_matched}/{target_total}"
                            eval_result["is_realtime"] = True

                            if 'word_results' in eval_result and isinstance(eval_result['word_results'], list):
                                for w_res in eval_result['word_results']:
                                    w_idx = w_res.get('index')
                                    if w_idx in session_matched_indices and w_res.get('status') == 'unread':
                                        w_res['status'] = 'matched'

                            if target_total > 0 and matched_words >= target_total:
                                is_auto_completed_sent = True
                                dynamic_tokens = max(32, min(256, target_total * 6 + 30))
                                final_audio = bytes(audio_pcm_buffer)
                                final_raw, final_conf = TarteelASR.transcribe_pcm(final_audio, max_tokens=dynamic_tokens)

                                chosen_raw = final_raw if (final_raw and len(final_raw.split()) >= len(ordered_words)) else accumulated_text
                                if not chosen_raw:
                                    chosen_raw = clean

                                final_eval = MakhrajEngine.evaluate_realtime_stream(
                                    target_ayah_text=target_ayah_text,
                                    recognized_speech_text=chosen_raw
                                )
                                final_eval["status"] = "auto_completed"
                                final_eval["raw_transcript"] = chosen_raw
                                final_eval["recognized_speech_text"] = chosen_raw
                                final_eval["source"] = "tarteel"
                                final_eval["model_confidence"] = round(final_conf if final_conf > 0 else 0.88, 3)
                                eval_result = final_eval
                            else:
                                eval_result["status"] = "evaluating"
                                live_accumulated = accumulated_text if accumulated_text else clean
                                eval_result["raw_transcript"] = live_accumulated
                                eval_result["recognized_speech_text"] = live_accumulated

                            logger.info(
                                f"⏱️ [WebSocket Stream] ASR: {t_asr:.3f}s | Status ({eval_result['status']}): "
                                f"'{eval_result.get('raw_transcript', clean)}' -> Match: {actual_matched}/{target_total} ({eval_result.get('accuracy')}%)"
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
