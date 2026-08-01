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
from iqraeval_asr import IqraEvalASR

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TalaqqiAIServer")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ASR models secara aman saat startup server."""
    logger.info("Server starting up — pre-loading IqraEval Wav2Vec2-300M model...")
    IqraEvalASR.preload_model()
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
    iqraeval_ready = IqraEvalASR.is_available()
    return {
        "server": "Smart Talaqqi AI Server (IqraEval 300M Exclusive)",
        "status": "healthy" if iqraeval_ready else "loading",
        "primary_asr": "IqraEval Wav2Vec2-300M",
        "iqraeval_ready": iqraeval_ready,
        "features": [
            "Phoneme-Level Mispronunciation Detection (IqraEval 2026)",
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
    WebSocket Real-Time Audio Stream Endpoint (Eksklusif IqraEval Wav2Vec2).
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
                        "iqraeval_ready": IqraEvalASR.is_available(),
                        "message": "Server AI Siap menerima stream audio murni IqraEval."
                    })

                elif action == "finish":
                    if target_ayah_text:
                        if len(audio_pcm_buffer) > 0 and IqraEvalASR.is_available():
                            raw_transcript, confidence = IqraEvalASR.transcribe_pcm(bytes(audio_pcm_buffer))
                            final_result = MakhrajEngine.evaluate_realtime_stream(
                                target_ayah_text=target_ayah_text,
                                recognized_speech_text=raw_transcript
                            )
                            final_result["source"] = "iqraeval"
                            final_result["status"] = "completed"
                            final_result["model_confidence"] = round(confidence, 3)
                            logger.info(f"[WebSocket Finish] Evaluasi Selesai -> Accuracy: {final_result.get('accuracy')}%, Transkrip Fonem: '{raw_transcript}', Makhraj Errors: {len(final_result.get('makhraj_errors', []))}, Model Conf: {confidence:.2%}")
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

                    if IqraEvalASR.is_available():
                        raw_transcript, confidence = IqraEvalASR.transcribe_pcm(
                            audio_window, return_confidence=False
                        )
                        # Filter IqraEval special tokens (<, <s>, <pad>, </s>, <unk>, |)
                        # yang lolos ke UI sebagai token kosong & bikin total_words=0
                        clean_tokens = [
                            t for t in (raw_transcript or "").split()
                            if t and not t.startswith("<") and t not in {"|", "<pad>", "<s>", "</s>", "<unk>"}
                        ]
                        if not clean_tokens:
                            continue
                        clean_transcript = " ".join(clean_tokens)
                        if not clean_transcript:
                            continue

                        # Evaluasi phoneme matching dengan target ayat (pakai transkrip bersih)
                        eval_result = MakhrajEngine.evaluate_realtime_stream(
                            target_ayah_text=target_ayah_text,
                            recognized_speech_text=clean_transcript
                        )
                        eval_result["status"] = "evaluating"
                        eval_result["source"] = "iqraeval"
                        eval_result["model_confidence"] = round(confidence, 3)
                        eval_result["raw_transcript"] = clean_transcript

                        # Pakai partial_accuracy dari engine (matched / panjang transkrip aktual) — JUJUR
                        target_total = eval_result.get('total_words') or eval_result.get('total_phonemes', 0) or 0
                        rec_total = eval_result.get('rec_phoneme_count', 0)
                        eval_result["partial_accuracy"] = eval_result.get('partial_accuracy', 0)
                        eval_result["progress"] = f"{min(rec_total, target_total)}/{target_total}"
                        eval_result["transcript_length"] = rec_total
                        eval_result["is_realtime"] = True

                        logger.info(
                            f"[WebSocket Stream] IqraEval Fonem: '{raw_transcript}' "
                            f"-> Match: {eval_result.get('matched_count')}/{target_total} "
                            f"| PartialAcc: {eval_result['partial_accuracy']}% "
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
