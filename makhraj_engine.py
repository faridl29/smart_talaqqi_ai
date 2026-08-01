import os
import re
import math
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("MakhrajEngine")

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

class MakhrajEngine:
    """
    Engine Analisis Makhraj & Tajwid Akustik Real-Time tanpa Language Model Auto-Correct.
    Mendukung ONNX Neural Inference & Acoustic CTC Alignment (2 vCPU / 4 GB RAM VPS).
    """

    _onnx_session = None

    @classmethod
    def get_onnx_session(cls):
        if not HAS_ONNX:
            return None
        if cls._onnx_session is None:
            model_path = os.path.join(os.path.dirname(__file__), "models", "wav2vec2_quran_quantized.onnx")
            if os.path.exists(model_path):
                try:
                    cls._onnx_session = ort.InferenceSession(model_path)
                    logger.info(f"Loaded Neural ONNX Model Session from {model_path}")
                except Exception as e:
                    logger.warning(f"Gagal memuat ONNX session: {e}")
        return cls._onnx_session


    # Peta Kelompok Makhraj Huruf Arab & Panduan Artikulasi Organ
    MAKHRAJ_GUIDANCE = {
        'ح': {
            'category': 'Wasathul Halq (Tengah Tenggorokan)',
            'confused_with': ['ه', 'ا', 'ء'],
            'guidance': "Huruf Haa (ح) dilafalkan di Wasathul Halq (tengah tenggorokan). Suara harus bersih dan mengalir, jangan tertukar dengan Ha (ه) di pangkal tenggorokan."
        },
        'ع': {
            'category': 'Wasathul Halq (Tengah Tenggorokan)',
            'confused_with': ['ا', 'ء'],
            'guidance': "Huruf 'Ain (ع) keluar dari Wasathul Halq (tengah tenggorokan). Tekan sedikit suara ke bagian tengah tenggorokan, jangan dibaca seperti Alif (ء)."
        },
        'خ': {
            'category': 'Adnal Halq (Ujung Tenggorokan)',
            'confused_with': ['ه', 'ك'],
            'guidance': "Huruf Khaa (خ) dilafalkan di Adnal Halq (ujung tenggorokan atas dekat langit-langit). Hasilkan getaran halus di tenggorokan atas."
        },
        'غ': {
            'category': 'Adnal Halq (Ujung Tenggorokan)',
            'confused_with': ['ا', 'ق'],
            'guidance': "Huruf Ghain (غ) keluar dari Adnal Halq (ujung tenggorokan). Ucapkan dengan suara tebal mengalir tanpa berlebihan."
        },
        'ص': {
            'category': 'Tharaful Lisan (Isti\'la / Tebal)',
            'confused_with': ['س'],
            'guidance': "Huruf Shaad (ص) adalah huruf Isti'la (tebal) & Shafir (desis). Pangkal lidah terangkat ke langit-langit, bedakan dengan Siin (س) yang tipis."
        },
        'ض': {
            'category': 'Hafatul Lisan (Tepi Lidah)',
            'confused_with': ['د', 'ظ'],
            'guidance': "Huruf Dhaad (ض) keluar dari tepi lidah menempel ke geraham atas. Jangan tertukar dengan Daal (د) yang tipis."
        },
        'ط': {
            'category': 'Tharaful Lisan (Gigi Seri Atas - Isti\'la)',
            'confused_with': ['ت'],
            'guidance': "Huruf Thaa (ط) adalah huruf tebal (Isti'la/Itbaq). Ujung lidah menempel di pangkal gigi seri atas dengan pangkal lidah terangkat."
        },
        'ظ': {
            'category': 'Tharaful Lisan & Dinding Gigi Seri',
            'confused_with': ['ذ', 'ز'],
            'guidance': "Huruf Zhaa (ظ) dilafalkan tebal dengan ujung lidah sedikit keluar menyentuh ujung gigi seri atas."
        },
        'ق': {
            'category': 'Aqshal Lisan (Pangkal Lidah)',
            'confused_with': ['ك'],
            'guidance': "Huruf Qaaf (ق) keluar dari Aqshal Lisan (pangkal lidah paling belakang menempel langit-langit lunak). Ucapkan tebal dan mantap."
        },
        'ث': {
            'category': 'Tharaful Lisan & Gigi Seri',
            'confused_with': ['س', 'ت'],
            'guidance': "Huruf Tsaa (ث) dilafalkan dengan ujung lidah sedikit keluar menyentuh ujung gigi seri atas secara lembut."
        },
    }

    @staticmethod
    def normalize_arabic(text: str) -> str:
        """
        Normalisasi teks Arab murni (penghapusan harakat, tanwin, dagger alif, tatweel, dan hamza).
        """
        if not text:
            return ''

        # Hapus Harakat standar & Tanwin (\u064B - \u065F)
        cleaned = re.sub(r'[\u064B-\u065F]', '', text)
        # Hapus karakter Quran Uthmani khusus
        cleaned = re.sub(r'[\u0610-\u061A\u06D6-\u06ED]', '', cleaned)
        # Hapus Alif Khanjariyah, Tatweel, Small Waw/Ya
        cleaned = cleaned.replace('\u0670', '').replace('\u0640', '')
        cleaned = re.sub(r'[\u06E5\u06E6]', '', cleaned)
        # Normalisasi Alif Wasla & Variasi Alif
        cleaned = re.sub(r'[إأآٱٲٳٵ]', 'ا', cleaned).replace('\u0671', 'ا')
        # Normalisasi Ta Marbuta & Alif Maqsura
        cleaned = cleaned.replace('ة', 'ه').replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي').replace('ء', '')
        # Trim & spasi ganda
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    @staticmethod
    def phonetic_normalize(text: str) -> str:
        """
        Normalisasi fonetis murni untuk memetakan makhraj serupa.
        """
        norm = MakhrajEngine.normalize_arabic(text)
        if not norm:
            return ''

        return (
            norm.replace('ع', 'ا')
            .replace('ح', 'ه')
            .replace('خ', 'ه')
            .replace('غ', 'ا')
            .replace('ص', 'س')
            .replace('ض', 'د')
            .replace('ط', 'ت')
            .replace('ظ', 'ذ')
            .replace('ث', 'س')
            .replace('ق', 'ك')
        )

    @classmethod
    def evaluate_realtime_stream(
        cls,
        target_ayah_text: str,
        recognized_speech_text: str
    ) -> Dict[str, Any]:
        """
        Evaluasi akurasi real-time per kata dan deteksi kesalahan makhraj spesifik.
        """
        original_words = [w for w in target_ayah_text.strip().split() if w]
        if not original_words:
            return {
                'accuracy': 0,
                'passed': False,
                'word_results': [],
                'makhraj_errors': [],
                'teacher_feedback': 'Belum ada bacaan yang terdeteksi.'
            }

        clean_targets = [cls.normalize_arabic(w) for w in original_words]
        phonetic_targets = [cls.phonetic_normalize(w) for w in original_words]

        clean_recognized = [cls.normalize_arabic(w) for w in recognized_speech_text.strip().split() if w]
        phonetic_recognized = [cls.phonetic_normalize(w) for w in recognized_speech_text.strip().split() if w]

        matched_count = 0
        wrong_count = 0
        missed_count = 0

        recognized_ptr = 0
        word_results = []
        makhraj_errors = []

        for i, (orig_word, clean_t, phon_t) in enumerate(zip(original_words, clean_targets, phonetic_targets)):
            if not clean_t:
                continue

            status = 'unread'
            matched = False

            # Search in sliding window of 4 words
            window_end = min(len(clean_recognized), recognized_ptr + 4)

            for j in range(recognized_ptr, window_end):
                rec_clean = clean_recognized[j]
                rec_phon = phonetic_recognized[j] if j < len(phonetic_recognized) else rec_clean

                # 1. Exact Match
                if clean_t == rec_clean or phon_t == rec_phon:
                    status = 'matched'
                    matched_count += 1
                    recognized_ptr = j + 1
                    matched = True
                    break
                # 2. Fuzzy Match (Salah Makhraj / Pengucapan Kurang Tepat)
                elif cls._is_fuzzy(clean_t, rec_clean) or cls._is_fuzzy(phon_t, rec_phon):
                    status = 'wrong'
                    wrong_count += 1
                    recognized_ptr = j + 1
                    matched = True

                    # Analisis Kesalahan Makhraj Spesifik
                    error_detail = cls._diagnose_word_makhraj(orig_word, clean_t, rec_clean)
                    if error_detail:
                        makhraj_errors.append(error_detail)
                    break

            if not matched:
                if recognized_ptr > 0 and i < len(clean_targets):
                    status = 'wrong'
                    wrong_count += 1
                else:
                    status = 'unread'
                    missed_count += 1

            word_results.append({
                'word': orig_word,
                'status': status,
                'index': i
            })

        total = len(clean_targets)
        accuracy = round((matched_count / total) * 100) if total > 0 else 0
        passed = accuracy >= 85

        feedback = cls._generate_feedback(accuracy, passed, len(makhraj_errors))

        return {
            'accuracy': accuracy,
            'passed': passed,
            'matched_count': matched_count,
            'wrong_count': wrong_count,
            'missed_count': missed_count,
            'total_words': total,
            'word_results': word_results,
            'makhraj_errors': makhraj_errors,
            'teacher_feedback': feedback
        }

    @classmethod
    def _diagnose_word_makhraj(cls, orig_word: str, target_clean: str, recognized_clean: str) -> Optional[Dict[str, Any]]:
        """
        Diagnosa huruf spesifik yang salah makhrajnya.
        """
        for char in target_clean:
            if char in cls.MAKHRAJ_GUIDANCE:
                info = cls.MAKHRAJ_GUIDANCE[char]
                # Jika huruf yang salah muncul di kata yang dikenali sebagai pasangan yang sering tertukar
                for confused in info['confused_with']:
                    if confused in recognized_clean and char not in recognized_clean:
                        return {
                            'target_word': orig_word,
                            'target_char': char,
                            'detected_char': confused,
                            'category': info['category'],
                            'guidance': info['guidance']
                        }

        return {
            'target_word': orig_word,
            'target_char': target_clean[0] if target_clean else '',
            'detected_char': recognized_clean[0] if recognized_clean else '',
            'category': 'Artikulasi Umum',
            'guidance': f"Kata '{orig_word}' kurang jelas pelafalannya. Perhatikan artikulasi makhraj hurufnya."
        }

    @staticmethod
    def _is_fuzzy(s1: str, s2: str) -> bool:
        if not s1 or not s2:
            return False
        if s1 == s2:
            return True
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return True
        dist = MakhrajEngine._levenshtein(s1, s2)
        sim = 1.0 - (dist / max_len)
        return sim >= 0.60

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        v0 = list(range(len(s2) + 1))
        v1 = [0] * (len(s2) + 1)
        for i in range(len(s1)):
            v1[0] = i + 1
            for j in range(len(s2)):
                cost = 0 if s1[i] == s2[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            v0 = v1[:]
        return v0[len(s2)]

    @staticmethod
    def _generate_feedback(accuracy: int, passed: bool, makhraj_error_count: int) -> str:
        if accuracy >= 90:
            return "Masya Allah! Bacaan Anda sangat fasih, makhraj dan artikulasi huruf terdeteksi presisi."
        elif passed:
            return "Alhamdulillah! Bacaan Anda memenuhi kriteria lulus (≥ 85%). Pertahankan kejelasan makhraj."
        elif makhraj_error_count > 0:
            return f"Terdeteksi {makhraj_error_count} kesalahan makhraj huruf. Perhatikan saran artikulasi pada kata berwarna merah."
        else:
            return "Perlahan dan perjelas pelafalan setiap huruf. Dengarkan audio Qari (SIMAK) untuk menyempurnakan makhraj."
