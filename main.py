import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from makhraj_engine import MakhrajEngine

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TalaqqiAIServer")

app = FastAPI(
    title="Smart Talaqqi AI Server",
    description="Real-Time WebSocket ASR & Makhraj AI Server untuk VPS Tencent (2 vCPU / 4 GB RAM)",
    version="1.0.0"
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
    return {
        "server": "Smart Talaqqi AI Server (Tencent VPS Ready)",
        "status": "healthy",
        "features": [
            "Real-Time WebSocket Audio Streaming",
            "Makhraj & Phonetic Error Diagnosis",
            "Zero Auto-Correct CTC Engine"
        ]
    }

@app.post("/api/v1/talaqqi/evaluate")
async def evaluate_recitation(req: RecitationRequest):
    """
    HTTP REST Endpoint untuk evaluasi bacaan & diagnosa makhraj.
    """
    try:
        result = MakhrajEngine.evaluate_realtime_stream(
            target_ayah_text=req.target_ayah_text,
            recognized_speech_text=req.recognized_speech_text
        )
        return result
    except Exception as e:
        logger.error(f"Error evaluating recitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/talaqqi/stream")
async def websocket_talaqqi_stream(websocket: WebSocket):
    """
    WebSocket Real-Time Audio & Transcript Stream Endpoint.
    Menerima potongan stream audio/transkrip real-time dari Flutter app,
    lalu membalas status kata & makhraj real-time dalam hitungan milidetik.
    """
    await websocket.accept()
    logger.info("Client WebSocket terkoneksi ke Talaqqi AI Server.")

    target_ayah_text = ""
    current_transcript = ""

    try:
        while True:
            # Terima pesan dari Flutter (JSON text)
            message = await websocket.receive_text()
            data: Dict[str, Any] = json.loads(message)

            action = data.get("action", "")

            if action == "init":
                target_ayah_text = data.get("target_ayah_text", "")
                current_transcript = ""
                logger.info(f"Init Talaqqi Stream untuk Target Ayat: '{target_ayah_text}'")
                await websocket.send_json({
                    "status": "initialized",
                    "message": "Server AI Siap menerima stream bacaan."
                })

            elif action == "stream_text" or action == "stream_audio":
                # Terima transkrip / fonem real-time dari client stream
                transcript_chunk = data.get("transcript", "")
                if transcript_chunk:
                    current_transcript = transcript_chunk

                # Evaluasi Makhraj Real-Time
                if target_ayah_text:
                    eval_result = MakhrajEngine.evaluate_realtime_stream(
                        target_ayah_text=target_ayah_text,
                        recognized_speech_text=current_transcript
                    )
                    eval_result["status"] = "evaluating"
                    await websocket.send_json(eval_result)

            elif action == "finish":
                # Evaluasi akhir saat user selesai membaca
                if target_ayah_text:
                    final_result = MakhrajEngine.evaluate_realtime_stream(
                        target_ayah_text=target_ayah_text,
                        recognized_speech_text=current_transcript
                    )
                    final_result["status"] = "completed"
                    await websocket.send_json(final_result)
                break

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
