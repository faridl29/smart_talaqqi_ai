"""
Smart Talaqqi AI Server — Makhraj & Phonetic Analysis Engine.

Pendekatan hybrid:
1) Nawar Halabi-style G2P Phonetiser (pure Python, zero-install) mengubah
   teks Arab ber-harakat (Al-Quran Uthmani) menjadi string fonem.
2) Tarteel Whisper Tiny ASR (tarteel-ai/whisper-tiny-ar-quran) memberikan
   teks Arab hasil transkripsi bacaan pengguna.
3) Kedua string fonem (target + recognized) dibandingkan dengan Levenshtein
   Distance + aligned-diff untuk deteksi makhraj / harakat / mad per-fonem.

Bekerja otomatis untuk SELURUH ayat Al-Quran tanpa definisi manual.
"""

import os
import re
import math
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("MakhrajEngine")


class MakhrajEngine:
    """Engine Analisis Makhraj Akustik Real-Time berbasis PHONEME MATCHING."""

    MAKHRAJ_GUIDANCE = {
        'ح': {
            'category': 'Wasathul Halq (Tengah Tenggorokan)',
            'guidance': "Huruf Haa (ح) keluar dari tengah tenggorokan. Suara harus bersih; jangan tertukar dengan Ha (ه) dari dasar tenggorokan."
        },
        'ع': {
            'category': 'Wasathul Halq (Tengah Tenggorokan)',
            'guidance': "Huruf 'Ain (ع) keluar dari tengah tenggorokan dengan sedikit tekanan; jangan dibaca seperti Alif (ا)."
        },
        'ه': {
            'category': 'Adnal Halq (Dasar Tenggorokan)',
            'guidance': "Huruf Ha (ه) dari dasar tenggorokan, hembusan napas halus."
        },
        'خ': {
            'category': 'Adnal Halq (Ujung Tenggorokan)',
            'guidance': "Huruf Khaa (خ) keluar dari ujung tenggorokan atas; getaran halus."
        },
        'غ': {
            'category': 'Adnal Halq (Ujung Tenggorokan)',
            'guidance': "Huruf Ghain (غ) keluar dari ujung tenggorokan; suara tebal mengalir."
        },
        'ص': {
            'category': "Isti'la / Tebal (Shafir)",
            'guidance': "Shaad (ص) huruf tebal & desis; pangkal lidah terangkat. Bedakan dengan Siin (س) yang tipis."
        },
        'ض': {
            'category': 'Hafatul Lisan (Tepi Lidah)',
            'guidance': "Dhaad (ض) dari tepi lidah menempel ke geraham atas; jangan dibaca tipis seperti Daal (د)."
        },
        'ط': {
            'category': "Isti'la / Tebal (Itbaq)",
            'guidance': "Thaa (ط) huruf tebal; ujung lidah di pangkal gigi seri atas, pangkal lidah terangkat."
        },
        'ظ': {
            'category': 'Tepi & Ujung Lidah',
            'guidance': "Zhaa (ظ) dilafalkan tebal dengan ujung lidah menyentuh gigi seri atas."
        },
        'ق': {
            'category': 'Aqshal Lisan (Pangkal Lidah)',
            'guidance': "Qaaf (ق) keluar dari pangkal lidah paling belakang; tebal dan mantap."
        },
        'ك': {
            'category': 'Wasatul Lisan',
            'guidance': "Kaf (ك) keluar dari tengah lidah; jangan tertukar dengan Qaaf (ق) yang lebih tebal."
        },
        'ث': {
            'category': 'Ujung Lidah & Gigi Seri',
            'guidance': "Tsaa (ث) dengan ujung lidah menyentuh gigi seri atas secara lembut."
        },
        'ذ': {
            'category': 'Ujung Lidah & Gigi Seri',
            'guidance': "Dzal (ذ) tipis; ujung lidah di antara gigi seri."
        },
    }

    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Normalisasi teks Arab murni (hapus harakat, kontrol, dll)."""
        if not text:
            return ''
        cleaned = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]', '', text)
        cleaned = re.sub(r'[\u0610-\u061A\u06D6-\u06ED]', '', cleaned)
        cleaned = re.sub(r'[\u064B-\u065F\u0670]', '', cleaned)
        cleaned = cleaned.replace('\u0640', '')
        cleaned = re.sub(r'[\u06E5\u06E6]', '', cleaned)
        cleaned = re.sub(r'[إأآٱٲٳٵ]', 'ا', cleaned).replace('\u0671', 'ا')
        cleaned = cleaned.replace('ة', 'ه').replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي').replace('ء', '')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    _ARABIC_TO_PHONEME = {
        'ا': '',
        'ب': 'b',
        'ت': 't',
        'ث': 'th',
        'ج': 'j',
        'ح': 'H',
        'خ': 'kh',
        'د': 'd',
        'ذ': 'dh',
        'ر': 'r',
        'ز': 'z',
        'س': 's',
        'ش': 'sh',
        'ص': 'S',
        'ض': 'D',
        'ط': 'T',
        'ظ': 'Z',
        'ع': 'E',
        'غ': 'gh',
        'ف': 'f',
        'ق': 'q',
        'ك': 'k',
        'ل': 'l',
        'م': 'm',
        'ن': 'n',
        'ه': 'h',
        'و': '',
        'ي': '',
        'ى': '',
        'ة': 'h',
    }

    _PHONEME_TO_ARABIC = {
        'b': 'ب', 't': 'ت', 'th': 'ث', 'j': 'ج',
        'H': 'ح', 'kh': 'خ', 'd': 'د', 'dh': 'ذ',
        'r': 'ر', 'z': 'ز', 's': 'س', 'sh': 'ش',
        'S': 'ص', 'D': 'ض', 'T': 'ط', 'Z': 'ظ',
        'E': 'ع', 'gh': 'غ', 'f': 'ف', 'q': 'ق',
        'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن',
        'h': 'ه',
    }

    @classmethod
    def arabic_word_to_phonemes(cls, word: str) -> List[str]:
        """Konversi satu kata Arab ber-harakat menjadi list token fonem."""
        if not word:
            return []

        s = word
        if s.startswith('\u0671'):
            s = 'ا' + s[1:]

        result: List[str] = []

        chars: List[Tuple[str, Optional[str]]] = []
        pending_shadda = False
        j = 0
        n = len(s)
        while j < n:
            c = s[j]
            if '\u064B' <= c <= '\u065F' or c in '\u0670\u0651\u0652':
                if c == '\u0651':
                    pending_shadda = True
                elif chars and chars[-1][1] is None:
                    chars[-1] = (chars[-1][0], c)
                else:
                    chars.append(('', c))
                j += 1
                continue
            elif '\u0600' <= c <= '\u06FF':
                if chars and chars[-1][1] is None and pending_shadda and chars[-1][0] == c:
                    chars[-1] = (c + c, chars[-1][1])
                    pending_shadda = False
                else:
                    chars.append((c, None))
                    if pending_shadda:
                        pending_shadda = False
                j += 1
                continue
            else:
                j += 1

        for letter, harakat in chars:
            if not letter:
                continue

            v = ''
            if harakat == '\u064E':
                v = 'a'
            elif harakat == '\u0650':
                v = 'i'
            elif harakat == '\u064F':
                v = 'u'

            is_shaddah = len(letter) == 2
            base = letter[0] if is_shaddah else letter

            if base == 'ا':
                result.append('aa')
                continue
            if base == 'و':
                if v == 'u':
                    result.append('uu')
                    continue
                else:
                    result.append('w')
                    continue
            if base == 'ى' or base == 'ي':
                if v == 'i':
                    result.append('ii')
                    continue
                else:
                    result.append('y')
                    continue

            ph = cls._ARABIC_TO_PHONEME.get(base, '')
            if not ph:
                continue

            if is_shaddah:
                result.append(ph)
                result.append('a')
                result.append(ph)
            else:
                result.append(ph)
                if v:
                    result.append(v)

        return result

    @classmethod
    def arabic_to_phonemes(cls, text_arabic: str) -> List[str]:
        """Konversi seluruh ayat Arab menjadi list token fonem."""
        if not text_arabic:
            return []
        result: List[str] = []
        for w in text_arabic.strip().split():
            if w:
                result.extend(cls.arabic_word_to_phonemes(w))
        return result

    @staticmethod
    def aligned_diff(s1: List[str], s2: List[str]) -> List[Tuple[str, str, str]]:
        n, m = len(s1), len(s2)
        if n == 0:
            return [('ins', '', t) for t in s2]
        if m == 0:
            return [('del', t, '') for t in s1]

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        ops = []
        i, j = n, m
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                ops.append(('equal', s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                ops.append(('sub', s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                ops.append(('del', s1[i - 1], ''))
                i -= 1
            else:
                ops.append(('ins', '', s2[j - 1]))
                j -= 1
        while i > 0:
            ops.append(('del', s1[i - 1], ''))
            i -= 1
        while j > 0:
            ops.append(('ins', '', s2[j - 1]))
            j -= 1

        ops.reverse()
        return ops

    @classmethod
    def evaluate_realtime_stream(
        cls,
        target_ayah_text: str,
        recognized_speech_text: str
    ) -> Dict[str, Any]:
        """Evaluasi akurat real-time berbasis phoneme matching."""
        target_tokens = cls.arabic_to_phonemes(target_ayah_text)
        if not target_tokens:
            return cls._empty_result('Tidak ada target ayat.')

        if not recognized_speech_text:
            rec_tokens = []
        elif any('\u0600' <= c <= '\u06FF' for c in recognized_speech_text):
            rec_tokens = cls.arabic_to_phonemes(recognized_speech_text)
        else:
            rec_tokens = []
            for tok in recognized_speech_text.strip().split():
                t = tok.replace('|', ' ').replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
                if t and t not in {'<pad>', '<s>', '</s>'}:
                    rec_tokens.extend(cls._normalize_iqraeval_token(t))

        if not rec_tokens:
            return cls._empty_result('Belum ada bacaan yang terdeteksi.')

        ops = cls.aligned_diff(target_tokens, rec_tokens)
        matched = sum(1 for op, _, _ in ops if op == 'equal')
        total = len(target_tokens)
        rec_total = len(rec_tokens)
        accuracy = round((matched / total) * 100) if total else 0
        # Fair partial: matched vs panjang transkrip aktual (bukan target penuh)
        partial_accuracy = round((matched / rec_total) * 100, 1) if rec_total else 0.0

        word_results, makhraj_errors = cls._analyze_word_level(
            target_ayah_text, target_tokens, rec_tokens, ops
        )

        passed = accuracy >= 85
        feedback = cls._generate_feedback(accuracy, passed, len(makhraj_errors))

        return {
            'accuracy': accuracy,
            'partial_accuracy': partial_accuracy,
            'rec_phoneme_count': rec_total,
            'passed': passed,
            'matched_count': matched,
            'total_phonemes': total,
            'matched_phonemes': matched,
            'total_words': total,  # alias backward-compat: Flutter pakai ini sbg 'target ayat'
            'word_results': word_results,
            'makhraj_errors': makhraj_errors,
            'teacher_feedback': feedback,
            'target_phonemes': ' '.join(target_tokens),
            'recognized_phonemes': ' '.join(rec_tokens),
        }

    # ─── Normalisasi Token IqraEval → Nawar Halabi Standard ───
    # IqraEval kadang mengeluarkan token non-standard seperti:
    #   'naay', 'ruuH', 'HII', 'mniin', 'aH', 'HII mniin', dll.
    # Token-token ini perlu dipecah menjadi unit fonem standard.
    _IQRAEVAL_FRAGMENTS = {
        # (prefix_sufix_combinations) → list of normalized tokens
        'ii': ['ii'],
        'uu': ['uu'],
        'aa': ['aa'],
        'aH': ['aa', 'H'],
        'iH': ['ii', 'H'],
        'uH': ['uu', 'H'],
        'all': ['a', 'l', 'l'],
        'alla': ['a', 'l', 'l', 'aa'],
        'allah': ['a', 'l', 'l', 'aa', 'h'],
        'lla': ['l', 'aa'],
        'llaa': ['l', 'aa'],
        'llah': ['l', 'aa', 'h'],
        'ill': ['i', 'l'],
        'illh': ['i', 'l', 'h'],
        'illah': ['i', 'l', 'aa', 'h'],
        'rabbi': ['r', 'a', 'b'],
        'rabbil': ['r', 'a', 'b', 'i', 'l'],
        'rbb': ['r', 'b'],
        'Huu': ['H', 'uu'],
        'Haa': ['H', 'aa'],
        'raH': ['r', 'a', 'H'],
        'raaH': ['r', 'aa', 'H'],
        'Hmaan': ['H', 'm', 'aa', 'n'],
        'maan': ['m', 'aa', 'n'],
        'Hii': ['H', 'ii'],
        'HII': ['H', 'ii'],
        'HIIm': ['H', 'ii', 'm'],
        'Hii m': ['H', 'ii', 'm'],
        'mniin': ['m', 'ii', 'n'],
        'miin': ['m', 'ii', 'n'],
        'mii n': ['m', 'ii', 'n'],
        'aamin': ['aa', 'm', 'ii', 'n'],
        'aali m': ['aa', 'l', 'ii', 'm'],
        'aaliim': ['aa', 'l', 'ii', 'm'],
    }

    # Pemetaan konsonan Nawar Halabi → IqraEval variants
    # (IqraEval uppercase H untuk ح, T untuk ط, dll)
    _CONSONANT_ALIASES = {
        'b': 'b', 't': 't', 'th': 'th', 'j': 'j',
        'H': 'H', 'kh': 'kh', 'd': 'd', 'dh': 'dh',
        'r': 'r', 'z': 'z', 's': 's', 'sh': 'sh',
        'S': 'S', 'D': 'D', 'T': 'T', 'Z': 'Z',
        'E': 'E', 'gh': 'gh', 'f': 'f', 'q': 'q',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'h': 'h',
    }

    @classmethod
    def _normalize_iqraeval_token(cls, token: str) -> List[str]:
        """
        Pecah token IqraEval (e.g. 'naay', 'ruuH', 'HII', 'mniin') ke fonem standard.

        Strategi greedy: split dari kiri ke kanan, cocokkan longest-substring
        yang merupakan konsonan IqraEval ATAU vokal pendek/panjang (aa/ii/uu),
        dengan fallback ke pemecahan per-karakter.
        """
        if not token:
            return []

        # Build reverse lookup: IqraEval consonant token → Nawar Halabi token
        rev: Dict[str, str] = {}
        for nh_token, variants in {
            'b': ['b'], 't': ['t'], 'th': ['th'], 'j': ['j'],
            'H': ['H'], 'kh': ['kh'], 'd': ['d'], 'dh': ['dh'],
            'r': ['r'], 'z': ['z'], 's': ['s'], 'sh': ['sh'],
            'S': ['S'], 'D': ['D'], 'T': ['T'], 'Z': ['Z'],
            'E': ['E'], 'gh': ['gh'], 'f': ['f'], 'q': ['q'],
            'k': ['k'], 'l': ['l'], 'm': ['m'], 'n': ['n'], 'h': ['h'],
        }.items():
            for v in variants:
                rev[v] = nh_token

        # Kamus compound pattern
        compounds = {
            'aa': 'aa', 'ii': 'ii', 'uu': 'uu',
            'aH': ['aa', 'H'], 'iH': ['ii', 'H'], 'uH': ['uu', 'H'],
            'aah': ['aa', 'h'], 'iih': ['ii', 'h'],
            'aam': ['aa', 'm'], 'iim': ['ii', 'm'],
        }

        # 1) Cek exact match di kamus fragment (handles 'all', 'illah', etc.)
        if token in cls._IQRAEVAL_FRAGMENTS:
            return list(cls._IQRAEVAL_FRAGMENTS[token])

        # 2) Greedy longest-substring matching
        normalized: List[str] = []
        s = token
        i = 0
        n = len(s)

        # Konversi string ke lowercase helper untuk konsonan (kecuali H, S, D, T, Z, E)
        cons_set = set(rev.keys())

        while i < n:
            matched = False
            # Cek 7-char dulu (terpanjang)
            for length in range(min(7, n - i), 0, -1):
                sub = s[i:i + length]

                # Cek compound pattern
                if sub in compounds:
                    val = compounds[sub]
                    if isinstance(val, list):
                        normalized.extend(val)
                    else:
                        normalized.append(val)
                    i += length
                    matched = True
                    break

                # Cek konsonan IqraEval
                if sub in cons_set:
                    normalized.append(rev[sub])
                    i += length
                    matched = True
                    break

                # Cek vokal pendek 'a', 'i', 'u'
                if sub in {'a', 'i', 'u'}:
                    normalized.append(sub)
                    i += length
                    matched = True
                    break

            if not matched:
                # Coba single char fallback (case-insensitive konsonan)
                ch = s[i].lower()
                if ch in cons_set:
                    normalized.append(rev[ch])
                elif s[i] in 'aAiIuU':
                    # Huruf vokal (rare in IqraEval since they're separate)
                    if s[i] == 'A':
                        normalized.append('a')
                    elif s[i] == 'I':
                        normalized.append('i')
                    elif s[i] == 'U':
                        normalized.append('u')
                    else:
                        normalized.append(s[i].lower())
                # else: karakter tak dikenal → skip
                i += 1

        return normalized

    @classmethod
    def _analyze_word_level(
        cls,
        original_text: str,
        target_tokens: List[str],
        rec_tokens: List[str],
        ops: List[Tuple[str, str, str]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        words_arabic = [w for w in original_text.strip().split() if w]
        word_token_counts: List[int] = []
        for aw in words_arabic:
            n = len(cls.arabic_word_to_phonemes(aw))
            word_token_counts.append(n)

        target_token_to_word: List[int] = []
        for w_idx, count in enumerate(word_token_counts):
            target_token_to_word.extend([w_idx] * count)

        word_matched: Dict[int, int] = {i: 0 for i in range(len(words_arabic))}
        word_total: Dict[int, int] = {i: 0 for i in range(len(words_arabic))}
        word_errors: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(words_arabic))}

        target_idx = 0
        rec_idx = 0
        for op, t_tok, r_tok in ops:
            if op == 'equal':
                word_idx = target_token_to_word[target_idx] if target_idx < len(target_token_to_word) else -1
                if word_idx >= 0:
                    word_matched[word_idx] = word_matched.get(word_idx, 0) + 1
                    word_total[word_idx] = word_total.get(word_idx, 0) + 1
                target_idx += 1
                rec_idx += 1
            elif op == 'sub':
                word_idx = target_token_to_word[target_idx] if target_idx < len(target_token_to_word) else -1
                if word_idx >= 0:
                    word_total[word_idx] = word_total.get(word_idx, 0) + 1
                    if r_tok in {'a', 'i', 'u', 'aa', 'ii', 'uu'} and t_tok in {'a', 'i', 'u', 'aa', 'ii', 'uu'}:
                        if t_tok in {'aa', 'ii', 'uu'} or r_tok in {'aa', 'ii', 'uu'}:
                            word_errors[word_idx].append({
                                'type': 'mad',
                                'target': t_tok,
                                'detected': r_tok,
                                'category': 'Kesalahan Mad (Durasi)',
                                'guidance': f"Vokal '{t_tok}' terbaca '{r_tok}'. Perhatikan kadar mad (2 harakat)."
                            })
                        else:
                            word_errors[word_idx].append({
                                'type': 'harakat',
                                'target_vowel': t_tok,
                                'detected_vowel': r_tok,
                                'category': 'Kesalahan Harakat (Vokal Pendek)',
                                'guidance': f"Vokal '{t_tok}' terbaca sebagai '{r_tok}'."
                            })
                    else:
                        arabic_char = cls._PHONEME_TO_ARABIC.get(t_tok, t_tok)
                        detected_char = cls._PHONEME_TO_ARABIC.get(r_tok, r_tok)
                        guidance = cls.MAKHRAJ_GUIDANCE.get(arabic_char, {}).get(
                            'guidance',
                            f"Fonem '{t_tok}' terbaca '{r_tok}'. Periksa makhraj."
                        )
                        category = cls.MAKHRAJ_GUIDANCE.get(arabic_char, {}).get('category', 'Artikulasi Makhraj')
                        word_errors[word_idx].append({
                            'type': 'makhraj',
                            'target_char': arabic_char,
                            'detected_char': detected_char,
                            'category': category,
                            'guidance': guidance,
                        })
                target_idx += 1
                rec_idx += 1
            elif op == 'ins':
                rec_idx += 1
            elif op == 'del':
                word_idx = target_token_to_word[target_idx] if target_idx < len(target_token_to_word) else -1
                if word_idx >= 0:
                    word_total[word_idx] = word_total.get(word_idx, 0) + 1
                    if t_tok not in {'a', 'i', 'u', 'aa', 'ii', 'uu'}:
                        arabic_char = cls._PHONEME_TO_ARABIC.get(t_tok, t_tok)
                        word_errors[word_idx].append({
                            'type': 'makhraj_missing',
                            'target_char': arabic_char,
                            'detected_char': '',
                            'category': 'Huruf Tidak Terdeteksi',
                            'guidance': f"Huruf '{arabic_char}' tidak terdengar. Pastikan dibaca dengan lengkap."
                        })
                target_idx += 1

        word_results = []
        for i, arabic_word in enumerate(words_arabic):
            matched = word_matched.get(i, 0)
            total = word_total.get(i, 0)
            if total == 0:
                status = 'unread'
            elif matched == total:
                status = 'matched'
            else:
                status = 'wrong'
            word_results.append({
                'word': arabic_word,
                'status': status,
                'index': i,
                'matched_phonemes': matched,
                'total_phonemes': total,
            })

        makhraj_errors = []
        for w_idx, errs in word_errors.items():
            for err in errs:
                err_with_word = dict(err)
                err_with_word['target_word'] = words_arabic[w_idx]
                makhraj_errors.append(err_with_word)

        return word_results, makhraj_errors

    @classmethod
    def _empty_result(cls, msg: str) -> Dict[str, Any]:
        return {
            'accuracy': 0,
            'passed': False,
            'matched_count': 0,
            'total_words': 0,
            'word_results': [],
            'makhraj_errors': [],
            'teacher_feedback': msg,
        }

    @staticmethod
    def _generate_feedback(accuracy: int, passed: bool, error_count: int) -> str:
        if passed:
            if accuracy == 100:
                return "Masha Allah, bacaan sempurna!"
            return f"Bagus ({accuracy}%). Perhatikan {error_count} detail kecil."
        if accuracy == 0:
            return "Tidak ada bacaan yang terdeteksi."
        if accuracy < 50:
            return f"Coba lagi dengan lebih teliti ({accuracy}%)."
        return f"Hampir ({accuracy}%). Perbaiki {error_count} kesalahan."
