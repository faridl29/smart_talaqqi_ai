"""
Quranic ASR — Tarteel Whisper Model via Native ONNX Runtime & PyTorch Fallback.

Optimized Native ONNX Engine for Real-Time Talaqqi (Apache-2.0 License).
- Zero PyTorch conversion overhead in ONNX mode (PCM -> numpy -> ONNX Runtime).
- Direct onnxruntime.InferenceSession with ONNX graph optimizations & KV Caching.
- Thread-safe singleton model loader (Double-Checked Locking).
- Low latency & low memory footprint tuned for 2 vCPU / 4 GB RAM VPS deployments.
"""

import logging
import os
import threading
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger("TarteelASR")

# Module availability flags
_ort_available: bool = False
_torch_available: bool = False

try:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    _ort_available = True
except Exception:
    _ort_available = False

try:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    _torch_available = True
except Exception:
    _torch_available = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TarteelASR:
    """
    Production-grade Wrapper for Tarteel Quran ASR.
    Supports Native ONNX Runtime ('eventhorizon0/tarteel-ai-onnx-whisper-base-ar-quran')
    with zero-Torch numpy pipeline and PyTorch CPU fallback.
    """

    MODEL_ID: str = os.getenv("MODEL_ID", "eventhorizon0/tarteel-ai-onnx-whisper-base-ar-quran")

    # Thread synchronization & Singleton state
    _lock: threading.Lock = threading.Lock()
    _is_loaded: bool = False
    _is_loading: bool = False
    _is_onnx: bool = False
    _warned_unavailable: bool = False

    # ONNX Sessions & Transformers Preprocessor
    _processor: Optional[Any] = None
    _encoder_sess: Optional[Any] = None
    _decoder_sess: Optional[Any] = None
    _decoder_past_sess: Optional[Any] = None
    
    # PyTorch Fallback Model
    _model: Optional[Any] = None

    # Pre-cached constant arrays & tokens for zero-allocation inference
    _prompt_tokens: List[int] = [50258, 50272, 50359, 50363]  # <|startoftranscript|> <|ar|> <|transcribe|> <|notimestamps|>
    _initial_input_ids: Optional[np.ndarray] = None
    _eos_token_id: int = 50257

    @classmethod
    def get_model(cls) -> Optional[object]:
        """
        Lazy-load Whisper Model (Native ONNX Runtime CPU / PyTorch CPU Fallback).
        Thread-safe singleton using Double-Checked Locking.
        """
        if not _ort_available and not _torch_available:
            if not cls._warned_unavailable:
                cls._warned_unavailable = True
                logger.warning("Neither onnxruntime nor torch/transformers are installed.")
            return None

        # Quick check without lock
        if cls._is_loaded:
            return cls._encoder_sess if cls._is_onnx else cls._model

        with cls._lock:
            # Double-check inside lock
            if cls._is_loaded:
                return cls._encoder_sess if cls._is_onnx else cls._model

            cls._is_loading = True
            try:
                from transformers import WhisperProcessor

                # 1. Native ONNX Runtime Engine Initialization
                if _ort_available:
                    try:
                        logger.info(f"Loading Native ONNX Model '{cls.MODEL_ID}' (ONNX Runtime CPU)...")
                        sub = 'onnx' if 'onnx' in cls.MODEL_ID.lower() else None
                        try:
                            cls._processor = WhisperProcessor.from_pretrained(cls.MODEL_ID, subfolder=sub)
                        except Exception:
                            cls._processor = WhisperProcessor.from_pretrained(cls.MODEL_ID)

                        # Locate model files
                        enc_path = hf_hub_download(cls.MODEL_ID, "encoder_model.onnx", subfolder="onnx")
                        dec_path = hf_hub_download(cls.MODEL_ID, "decoder_model.onnx", subfolder="onnx")
                        dec_past_path = hf_hub_download(cls.MODEL_ID, "decoder_with_past_model.onnx", subfolder="onnx")

                        # Configure session options for 2 vCPU / 4 GB RAM VPS
                        num_cores = max(1, min(2, os.cpu_count() or 2))
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

                        cls._is_onnx = True
                        cls._is_loaded = True
                        cls._model = cls._encoder_sess
                        logger.info(f"⚡ Native ONNX Engine '{cls.MODEL_ID}' loaded successfully! (Threads={num_cores})")
                        return cls._encoder_sess

                    except Exception as e_onnx:
                        logger.warning(f"Failed to load via Native ONNX Runtime ({e_onnx}), trying PyTorch fallback...")

                # 2. PyTorch Fallback Initialization
                if _torch_available:
                    logger.info(f"Loading HF PyTorch Model '{cls.MODEL_ID}' (PyTorch CPU Fallback)...")
                    cls._processor = WhisperProcessor.from_pretrained(cls.MODEL_ID)
                    cls._model = WhisperForConditionalGeneration.from_pretrained(cls.MODEL_ID)
                    cls._model.eval()
                    cls._is_onnx = False
                    cls._is_loaded = True

                    try:
                        num_cores = max(1, min(2, os.cpu_count() or 2))
                        torch.set_num_threads(num_cores)
                        logger.info(f"PyTorch threads set to {num_cores} for VPS CPU optimization.")
                    except Exception:
                        pass

                    logger.info(f"⚡ PyTorch Fallback Model '{cls.MODEL_ID}' loaded successfully!")
                    return cls._model

            except Exception as e:
                logger.error(f"Failed to load Tarteel ASR model: {e}")
                cls._model = None
                cls._processor = None
                cls._is_loaded = False
            finally:
                cls._is_loading = False

            return cls._model

    @classmethod
    def is_available(cls) -> bool:
        """Check if ASR model is loaded and ready for inference."""
        return (_ort_available or _torch_available) and cls.get_model() is not None

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

        Zero PyTorch overhead in ONNX mode:
        PCM -> float32 numpy -> ONNX Encoder -> ONNX Decoder (KV Cache Greedy Loop) -> Tokenizer

        Confidence Estimation:
        Calculated as the normalized geometric mean probability of all generated non-special tokens
        using log-softmax over output decoder logits:
        confidence = clip(exp(mean(log_softmax(logits[token]))), 0.0, 1.0)
        """
        model = cls.get_model()
        if model is None or cls._processor is None:
            return "", 0.0

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return "", 0.0

        try:
            # 1. PCM int16 -> float32 numpy [-1.0, 1.0] without torch
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if cls._is_onnx and cls._encoder_sess and cls._decoder_sess and cls._decoder_past_sess:
                # ── NATIVE ONNX RUNTIME INFERENCE (NO TORCH) ──
                inputs = cls._processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors="np",
                    return_attention_mask=False,
                )
                input_features = inputs.input_features.astype(np.float32)

                # Step 1: Run Encoder ONNX Session
                enc_out = cls._encoder_sess.run(None, {"input_features": input_features})
                encoder_hidden_states = enc_out[0]

                # Step 2: Run Initial Decoder Step
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

                # Construct Key-Value cache mapping for decoder_with_past
                past_kv: Dict[str, np.ndarray] = {}
                dec_out_names = [o.name for o in cls._decoder_sess.get_outputs()]
                for idx, name in enumerate(dec_out_names[1:]):
                    past_name = name.replace("present.", "past_key_values.")
                    past_kv[past_name] = dec_out[idx + 1]

                # Step 3: Run Decoder with Past KV Cache Loop
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

                    past_out_names = [o.name for o in cls._decoder_past_sess.get_outputs()]
                    for idx, name in enumerate(past_out_names[1:]):
                        past_name = name.replace("present.", "past_key_values.")
                        past_kv[past_name] = dec_past_out[idx + 1]

                # Step 4: Batch decode generated tokens
                text = cls._processor.batch_decode([tokens], skip_special_tokens=True)[0].strip()

                # Step 5: Compute Confidence Score
                confidence = 0.88
                if return_confidence and step_logprobs:
                    mean_logprob = float(np.mean(step_logprobs))
                    confidence = float(np.clip(np.exp(mean_logprob), 0.0, 1.0))

            else:
                # ── PYTORCH CPU FALLBACK (WHEN ONNX IS UNAVAILABLE) ──
                inputs = cls._processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                forced_decoder_ids = cls._processor.get_decoder_prompt_ids(language="ar", task="transcribe")
                tokenizer = cls._processor.tokenizer

                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "num_beams": 1,
                    "do_sample": False,
                    "forced_decoder_ids": forced_decoder_ids,
                    "eos_token_id": tokenizer.eos_token_id,
                    "no_repeat_ngram_size": 0,
                }

                with torch.inference_mode():
                    gen_out = cls._model.generate(
                        inputs.input_features,
                        attention_mask=inputs.attention_mask,
                        use_cache=True,
                        return_dict_in_generate=bool(return_confidence),
                        output_scores=bool(return_confidence),
                        **gen_kwargs
                    )

                sequences = gen_out.sequences if (return_confidence and hasattr(gen_out, 'sequences')) else gen_out
                text = cls._processor.batch_decode(
                    sequences, skip_special_tokens=True
                )[0].strip()

                confidence = 0.88
                if return_confidence and hasattr(gen_out, 'scores') and gen_out.scores:
                    try:
                        seq_ids = sequences[0]
                        step_logprobs = []
                        for step_logits, tok_id in zip(gen_out.scores, seq_ids[1:]):
                            lp = torch.log_softmax(step_logits[0], dim=-1)
                            step_logprobs.append(float(lp[tok_id].item()))
                        if step_logprobs:
                            confidence = float(np.clip(np.exp(np.mean(step_logprobs)), 0.0, 1.0))
                    except Exception:
                        pass

            engine_label = "Native ONNX" if cls._is_onnx else "PyTorch Fallback"
            logger.info(
                f"Tarteel Quran [{engine_label}] transcript ({len(pcm_bytes)} bytes, "
                f"conf={confidence:.2%}): '{text[:80]}{'...' if len(text) > 80 else ''}'"
            )
            return text, confidence

        except Exception as e:
            logger.error(f"Error transcribing Tarteel ASR: {e}")
            return "", 0.0
