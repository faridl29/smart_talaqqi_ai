"""
Quranic ASR — Tarteel Whisper Model via Pure Native ONNX Runtime.

Production-grade Native ONNX Engine for Real-Time Talaqqi (Apache-2.0 License).
- Pure ONNX Runtime execution (no PyTorch dependencies or fallbacks).
- PCM -> numpy -> ONNX Runtime Encoder/Decoder with KV Caching.
- Thread-safe singleton model loader (Double-Checked Locking).
- Ultra-low latency & memory footprint optimized for 2 vCPU / 4 GB RAM VPS deployments.
"""

import logging
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/root/.cache/huggingface"

import threading
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger("TarteelASR")

_ort_available: bool = False
try:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    _ort_available = True
except Exception:
    _ort_available = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TarteelASR:
    """
    Pure ONNX Runtime Wrapper for Tarteel Quran ASR.
    Model: 'eventhorizon0/tarteel-ai-onnx-whisper-base-ar-quran'
    """

    MODEL_ID: str = os.getenv("MODEL_ID", "eventhorizon0/tarteel-ai-onnx-whisper-base-ar-quran")

    # Thread synchronization & Singleton state
    _lock: threading.Lock = threading.Lock()
    _is_loaded: bool = False
    _is_loading: bool = False
    _warned_unavailable: bool = False

    # ONNX Sessions & Preprocessor
    _processor: Optional[Any] = None
    _encoder_sess: Optional[ort.InferenceSession] = None
    _decoder_sess: Optional[ort.InferenceSession] = None
    _decoder_past_sess: Optional[ort.InferenceSession] = None

    # Pre-cached static prompt tokens & input array
    _prompt_tokens: List[int] = [50258, 50272, 50359, 50363]  # <|startoftranscript|> <|ar|> <|transcribe|> <|notimestamps|>
    _initial_input_ids: Optional[np.ndarray] = None
    _eos_token_id: int = 50257

    @classmethod
    def get_model(cls) -> Optional[ort.InferenceSession]:
        """
        Lazy-load Whisper Model (Pure Native ONNX Runtime CPU).
        Thread-safe singleton using Double-Checked Locking.
        """
        if not _ort_available:
            if not cls._warned_unavailable:
                cls._warned_unavailable = True
                logger.error("onnxruntime is not installed. Native ONNX engine cannot start.")
            return None

        # Quick check without lock
        if cls._is_loaded:
            return cls._encoder_sess

        with cls._lock:
            # Double-check inside lock
            if cls._is_loaded:
                return cls._encoder_sess

            cls._is_loading = True
            try:
                from transformers import WhisperProcessor

                logger.info(f"Loading Pure Native ONNX Model '{cls.MODEL_ID}' (ONNX Runtime CPU)...")
                sub = 'onnx' if 'onnx' in cls.MODEL_ID.lower() else None
                try:
                    cls._processor = WhisperProcessor.from_pretrained(cls.MODEL_ID, subfolder=sub)
                except Exception:
                    cls._processor = WhisperProcessor.from_pretrained(cls.MODEL_ID)

                # Locate standard native ONNX model files from HuggingFace Hub
                enc_path = hf_hub_download(cls.MODEL_ID, "encoder_model.onnx", subfolder="onnx")
                dec_path = hf_hub_download(cls.MODEL_ID, "decoder_model.onnx", subfolder="onnx")
                dec_past_path = hf_hub_download(cls.MODEL_ID, "decoder_with_past_model.onnx", subfolder="onnx")

                # Apply ONNX Runtime Dynamic INT8 Quantization (50% RAM & CPU speedup)
                enable_int8 = os.getenv("ENABLE_INT8", "true").lower() in ("true", "1")
                enc_q_path = enc_path.replace(".onnx", "_dyn_quant.onnx")
                dec_q_path = dec_path.replace(".onnx", "_dyn_quant.onnx")
                dec_past_q_path = dec_past_path.replace(".onnx", "_dyn_quant.onnx")

                is_quantized = False
                if enable_int8:
                    try:
                        from onnxruntime.quantization import quantize_dynamic, QuantType
                        for src, dst in [(enc_path, enc_q_path), (dec_path, dec_q_path), (dec_past_path, dec_past_q_path)]:
                            if not os.path.exists(dst):
                                logger.info(f"Quantizing ONNX model for CPU INT8 -> '{os.path.basename(dst)}'...")
                                quantize_dynamic(src, dst, weight_type=QuantType.QUInt8)
                        enc_path, dec_path, dec_past_path = enc_q_path, dec_q_path, dec_past_q_path
                        is_quantized = True
                    except Exception as q_err:
                        logger.warning(f"ONNX Dynamic Quantization skipped: {q_err}. Using standard FP32 ONNX model.")
                else:
                    logger.info("ENABLE_INT8=false -> Using standard FP32 ONNX model.")

                # Configure CPU Session Options (uses all available cores)
                num_cores = max(1, os.cpu_count() or 2)
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = num_cores
                so.inter_op_num_threads = num_cores
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.enable_cpu_mem_arena = True
                so.enable_mem_pattern = True

                cls._encoder_sess = ort.InferenceSession(enc_path, sess_options=so)
                cls._decoder_sess = ort.InferenceSession(dec_path, sess_options=so)
                cls._decoder_past_sess = ort.InferenceSession(dec_past_path, sess_options=so)

                # Cache static prompt tokens and initial input array
                if hasattr(cls._processor, "tokenizer"):
                    tok = cls._processor.tokenizer
                    cls._eos_token_id = tok.eos_token_id or 50257
                    p_ids = cls._processor.get_decoder_prompt_ids(language="ar", task="transcribe")
                    cls._prompt_tokens = [50258] + [tid for _, tid in p_ids]

                cls._initial_input_ids = np.array([cls._prompt_tokens], dtype=np.int64)

                cls._is_loaded = True
                quant_str = " (Dynamic INT8 Quantized)" if is_quantized else " (FP32)"
                logger.info(f"⚡ Pure Native ONNX Engine '{cls.MODEL_ID}'{quant_str} loaded successfully! (Threads={num_cores})")
                return cls._encoder_sess

            except Exception as e:
                logger.error(f"Failed to load Native ONNX Tarteel ASR model: {e}")
                cls._encoder_sess = None
                cls._decoder_sess = None
                cls._decoder_past_sess = None
                cls._processor = None
                cls._is_loaded = False
            finally:
                cls._is_loading = False

            return cls._encoder_sess

    @classmethod
    def is_available(cls) -> bool:
        """Check if ONNX model is loaded and ready for inference."""
        return _ort_available and cls.get_model() is not None

    @classmethod
    def preload_model(cls) -> bool:
        """Pre-load model during server startup."""
        return cls.get_model() is not None

    @classmethod
    def transcribe_pcm(
        cls,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        return_confidence: bool = True,
        max_tokens: int = 32,
    ) -> Tuple[str, float]:
        """
        Transcribe raw PCM int16 mono audio -> (arabic_text_with_tashkeel, confidence_score).

        Pure ONNX Runtime Pipeline (Zero PyTorch Dependency):
        PCM int16 -> float32 numpy -> ONNX Encoder -> ONNX Decoder (KV Cache Greedy Loop) -> Tokenizer

        Confidence Estimation:
        Normalized geometric mean probability of all generated non-special tokens using log-softmax over output logits:
        confidence = clip(exp(mean(log_softmax(logits[token]))), 0.0, 1.0)
        """
        if not cls.is_available() or cls._encoder_sess is None or cls._decoder_sess is None or cls._decoder_past_sess is None or cls._processor is None:
            return "", 0.0

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return "", 0.0

        try:
            # 1. PCM int16 -> float32 numpy [-1.0, 1.0] without torch
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 2. Extract Log-Mel Spectrogram features into numpy ndarray
            inputs = cls._processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="np",
                return_attention_mask=False,
            )
            input_features = inputs.input_features.astype(np.float32)

            # 3. Encoder Inference
            enc_out = cls._encoder_sess.run(None, {"input_features": input_features})
            encoder_hidden_states = enc_out[0]  # shape: [1, 1500, 512]

            # 4. Initial Decoder Step
            initial_ids = cls._initial_input_ids if cls._initial_input_ids is not None else np.array([cls._prompt_tokens], dtype=np.int64)
            dec_out = cls._decoder_sess.run(None, {
                "input_ids": initial_ids,
                "encoder_hidden_states": encoder_hidden_states
            })

            logits = dec_out[0]  # shape: [1, seq_len, vocab_size]
            next_token = int(np.argmax(logits[0, -1, :]))

            step_logprobs: List[float] = []
            if return_confidence:
                l0 = logits[0, -1, :]
                max_l = float(np.max(l0))
                exp_l = np.exp(l0 - max_l)
                probs = exp_l / float(np.sum(exp_l))
                step_logprobs.append(float(np.log(probs[next_token] + 1e-12)))

            tokens: List[int] = list(cls._prompt_tokens) + [next_token]

            # Build Key-Value cache map for decoder_with_past
            past_kv: Dict[str, np.ndarray] = {}
            dec_out_names = [o.name for o in cls._decoder_sess.get_outputs()]
            for idx, name in enumerate(dec_out_names[1:]):
                past_name = name.replace("present.", "past_key_values.")
                past_kv[past_name] = dec_out[idx + 1]

            # 5. Decoder Loop with Past KV Cache
            for step in range(max_tokens - 1):
                if next_token == cls._eos_token_id:
                    break

                feed_dict = {
                    "input_ids": np.array([[next_token]], dtype=np.int64),
                    **past_kv
                }
                dec_past_out = cls._decoder_past_sess.run(None, feed_dict)
                logits = dec_past_out[0]
                next_token = int(np.argmax(logits[0, -1, :]))

                if return_confidence and next_token != cls._eos_token_id:
                    l0 = logits[0, -1, :]
                    max_l = float(np.max(l0))
                    exp_l = np.exp(l0 - max_l)
                    probs = exp_l / float(np.sum(exp_l))
                    step_logprobs.append(float(np.log(probs[next_token] + 1e-12)))

                tokens.append(next_token)

                # Deteksi loop repetisi n-gram (3 token berurutan sama) untuk hentikan halusinasi Whisper
                if len(tokens) >= 6 and tokens[-1] == tokens[-2] == tokens[-3]:
                    logger.debug("Tarteel ASR: Early stop due to token repetition loop.")
                    break

                past_out_names = [o.name for o in cls._decoder_past_sess.get_outputs()]
                for idx, name in enumerate(past_out_names[1:]):
                    past_name = name.replace("present.", "past_key_values.")
                    past_kv[past_name] = dec_past_out[idx + 1]

            # 6. Decode tokens into Arabic text
            text = cls._processor.batch_decode([tokens], skip_special_tokens=True)[0].strip()

            # 7. Calculate Confidence Score
            confidence = 0.88
            if return_confidence and step_logprobs:
                mean_logprob = float(np.mean(step_logprobs))
                confidence = float(np.clip(np.exp(mean_logprob), 0.0, 1.0))

            logger.info(
                f"Tarteel Quran [Pure ONNX] transcript ({len(pcm_bytes)} bytes, "
                f"conf={confidence:.2%}): '{text[:80]}{'...' if len(text) > 80 else ''}'"
            )
            return text, confidence

        except Exception as e:
            logger.error(f"Error transcribing Tarteel ASR: {e}")
            return "", 0.0
