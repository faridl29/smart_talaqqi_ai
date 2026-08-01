"""
IqraEval Wav2Vec2 ASR Engine — Phoneme-level Qur'anic Mispronunciation Recognition.
Model: FatimahEmadEldin/wav2vec2-xls-r-300m-iqraeval (300M params fine-tuned for Iqra'Eval 2026).
"""

import logging
import tempfile
import os
from typing import Optional, Tuple

# Mencegah conflict OpenMP / threading segfault di macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger = logging.getLogger("IqraEvalASR")

# Lazy imports agar server tetap bisa start tanpa torch / transformers
_iqraeval_available = False
try:
    import torch
    torch.set_num_threads(1)
    import numpy as np
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    _iqraeval_available = True
except ImportError:
    pass



class IqraEvalASR:
    """
    Wrapper Wav2Vec2-XLS-R-300M (Iqra'Eval) untuk transkripsi fonem & makhraj Al-Qur'an.
    Singleton pattern — model dan processor di-load sekali saja secara lazy.
    """
    MODEL_ID = "FatimahEmadEldin/wav2vec2-xls-r-300m-iqraeval"
    
    _model: Optional[object] = None
    _processor: Optional[object] = None
    _is_loading: bool = False
    _warned_unavailable: bool = False

    @classmethod
    def get_model_and_processor(cls) -> Tuple[Optional[object], Optional[object]]:
        """Load model Wav2Vec2 & Processor (lazy, sekali saja)."""
        if not _iqraeval_available:
            if not cls._warned_unavailable:
                cls._warned_unavailable = True
                logger.warning(
                    "torch / transformers tidak terinstall. "
                    "Jalankan: pip install torch transformers torchaudio. "
                    "Server tetap berjalan dengan fallback ke Whisper / STT HP."
                )
            return None, None

        if cls._model is not None and cls._processor is not None:
            return cls._model, cls._processor

        if cls._is_loading:
            return None, None

        cls._is_loading = True
        try:
            logger.info(
                f"Loading Wav2Vec2 IqraEval model '{cls.MODEL_ID}' "
                f"(CPU) — memuat model ~1.2 GB..."
            )
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(cls.MODEL_ID)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
            cls._processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            cls._model = Wav2Vec2ForCTC.from_pretrained(cls.MODEL_ID)
            cls._model.eval()  # Set to evaluation mode
            logger.info(
                f"Model '{cls.MODEL_ID}' berhasil dimuat ke memory."
            )

        except Exception as e:
            logger.error(f"Gagal memuat model IqraEval: {e}")
            cls._model = None
            cls._processor = None
        finally:
            cls._is_loading = False

        return cls._model, cls._processor

    @classmethod
    def is_available(cls) -> bool:
        """Cek apakah IqraEval ASR tersedia dan model sudah dimuat."""
        return _iqraeval_available and cls._model is not None and cls._processor is not None

    @classmethod
    @torch.inference_mode()
    def transcribe_pcm(
        cls,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        return_confidence: bool = True,
    ) -> Tuple[str, float]:
        """
        Transkripsi raw PCM 16kHz 16-bit mono → (token_string, confidence_score).

        Returns:
            Tuple (phoneme_string, confidence). Confidence ∈ [0,1] adalah
            rata-rata max-probability softmax dari logits model pada
            frame-frame yang diprediksi. Confidence rendah = noise / TV /
            ambient yang tidak dapat di-decode dengan pasti oleh model itu sendiri.
        """
        model, processor = cls.get_model_and_processor()
        if model is None or processor is None:
            return "", 0.0

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return "", 0.0

        try:
            import torch
            import torch.nn.functional as F
            import numpy as np

            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            inputs = processor(
                audio_np,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits = model(inputs.input_values).logits

            # Fast path: skip softmax + token-level iteration kalau tidak diminta
            if return_confidence:
                probs = F.softmax(logits, dim=-1)
                max_probs, predicted_ids = torch.max(probs, dim=-1)
                transcription_list = processor.batch_decode(predicted_ids)
                result = transcription_list[0] if transcription_list else ""

                predicted_ids_list = predicted_ids[0].tolist()
                valid_max_probs = []
                for token_id, p in zip(predicted_ids_list, max_probs[0].tolist()):
                    if token_id != 0:
                        valid_max_probs.append(p)
                confidence = sum(valid_max_probs) / len(valid_max_probs) if valid_max_probs else 0.0
            else:
                # Streaming mode: confidence murah dari logit margin (top1 - top2), tanpa softmax (~5% overhead)
                top2_logits, top2_ids = torch.topk(logits, k=2, dim=-1)
                margin = (top2_logits[..., 0] - top2_logits[..., 1]).clamp(min=0)
                ids_top1 = top2_ids[..., 0]
                transcription_list = processor.batch_decode(ids_top1)
                result = transcription_list[0] if transcription_list else ""
                # Normalisasi margin pakai sigmoid-ish: confidence = margin / (margin + 5)
                margin_list = margin[0].tolist()
                ids_list = ids_top1[0].tolist()
                valid_margins = [m for tid, m in zip(ids_list, margin_list) if tid != 0]
                if valid_margins:
                    avg_margin = sum(valid_margins) / len(valid_margins)
                    confidence = float(avg_margin / (avg_margin + 5.0))
                else:
                    confidence = 0.0

            logger.info(
                f"IqraEval Phoneme Transcript ({len(pcm_bytes)} bytes audio, "
                f"confidence={confidence:.2%}): "
                f"'{result[:80]}{'...' if len(result) > 80 else ''}'"
            )
            return result, confidence

        except Exception as e:
            logger.error(f"Error transkripsi IqraEval ASR: {e}")
            return "", 0.0

    @classmethod
    def preload_model(cls) -> bool:
        """
        Pre-load model saat server startup.
        Return True jika model berhasil dimuat.
        """
        model, processor = cls.get_model_and_processor()
        return model is not None and processor is not None
