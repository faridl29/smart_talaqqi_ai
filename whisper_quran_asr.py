"""
Quranic ASR — Tarteel Whisper Tiny via HuggingFace transformers.
API:
    transcribe_pcm(pcm_bytes, sample_rate=16000) -> (text, confidence)

Output: Arabic text dengan tashkeel, dipakai MakhrajEngine.arabic_to_phonemes.
"""

import logging
from typing import Tuple

logger = logging.getLogger("TarteelASR")

_qw_available = False
try:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    _qw_available = True
except ImportError:
    pass


class TarteelASR:
    """
    Wrapper transformers + tarteel-ai/whisper-tiny-ar-quran.
    Tiny (39M params) — CPU-friendly, fine-tuned khusus Quran.
    License: Apache-2.0 (model), Quran data MIT/Tarteel.
    """

    MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"

    _model = None
    _processor = None
    _is_loading = False
    _warned_unavailable = False

    @classmethod
    def get_model(cls) -> object:
        """Lazy-load HF Whisper Tiny (PyTorch, fp32). Singleton."""
        if not _qw_available:
            if not cls._warned_unavailable:
                cls._warned_unavailable = True
                logger.warning(
                    "transformers/torch tidak terinstall. "
                    "Jalankan: pip install transformers torch"
                )
            return None

        if cls._model is not None and cls._processor is not None:
            return cls._model

        if cls._is_loading:
            return None

        cls._is_loading = True
        # ponytail: global lock; per-callsite locks if multi-stream.
        try:
            logger.info(f"Loading HF Tarteel '{cls.MODEL_ID}' (cpu, fp32)...")
            cls._processor = WhisperProcessor.from_pretrained(cls.MODEL_ID)
            cls._model = WhisperForConditionalGeneration.from_pretrained(cls.MODEL_ID)
            cls._model.eval()
            # Force Arabic, suppress other langs
            try:
                cls._model.config.forced_decoder_ids = cls._processor.get_decoder_prompt_ids(
                    language="ar", task="transcribe"
                )
            except Exception:
                pass
            logger.info(f"Model '{cls.MODEL_ID}' loaded.")
        except Exception as e:
            logger.error(f"Gagal memuat model Tarteel: {e}")
            cls._model = None
            cls._processor = None
        finally:
            cls._is_loading = False

        return cls._model

    @classmethod
    def is_available(cls) -> bool:
        return _qw_available and cls._model is not None and cls._processor is not None

    @classmethod
    def transcribe_pcm(
        cls,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        return_confidence: bool = True,
    ) -> Tuple[str, float]:
        """
        Transcribe raw PCM int16 mono -> (arabic_text_with_tashkeel, confidence).

        Confidence: rata-rata token log-prob normalized ke [0, 1].
        """
        model = cls.get_model()
        if model is None or cls._processor is None:
            return "", 0.0

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return "", 0.0

        try:
            import numpy as np

            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            inputs = cls._processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )

            input_features = inputs.input_features
            attention_mask = inputs.attention_mask

            with torch.no_grad():
                gen_out = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = gen_out.sequences
            text = cls._processor.batch_decode(
                sequences, skip_special_tokens=True
            )[0].strip()

            confidence = 0.0
            if return_confidence and gen_out.scores:
                seq_ids = sequences[0]
                step_logprobs = []
                for step_logits, tok_id in zip(gen_out.scores, seq_ids[1:]):
                    lp = torch.log_softmax(step_logits[0], dim=-1)
                    step_logprobs.append(float(lp[tok_id].item()))
                if step_logprobs:
                    mean_lp = sum(step_logprobs) / len(step_logprobs)
                    # map [-1, 0] -> [0, 1]
                    confidence = float(max(0.0, min(1.0, 1.0 + mean_lp)))

            logger.info(
                f"Tarteel transcript ({len(pcm_bytes)} bytes, "
                f"conf={confidence:.2%}): '{text[:80]}{'...' if len(text) > 80 else ''}'"
            )
            return text, confidence

        except Exception as e:
            logger.error(f"Error transkripsi Tarteel ASR: {e}")
            return "", 0.0

    @classmethod
    def preload_model(cls) -> bool:
        return cls.get_model() is not None
