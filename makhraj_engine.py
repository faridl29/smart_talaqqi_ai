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
            'guidance': "Huruf Haa (ح) keluar dari tengah tenggorokan. Suara bersih mengalir halus tanpa hambatan dada.",
            'anatomy': "Tekan bagian tengah tenggorokan (epiglottis) secara halus ke dinding tenggorokan.",
            'tajweed_rule': 'Hams & Rakhawah (Desis & Mengalir)',
        },
        'ع': {
            'category': 'Wasathul Halq (Tengah Tenggorokan)',
            'guidance': "Huruf 'Ain (ع) keluar dari tengah tenggorokan dengan sedikit penekanan lisan.",
            'anatomy': "Penyempitan di tengah tenggorokan dengan pita suara bergetar sedang.",
            'tajweed_rule': 'Tawassut (Suara Sedang)',
        },
        'ه': {
            'category': 'Aqshal Halq (Dasar Tenggorokan)',
            'guidance': "Huruf Ha (ه) dari pangkal/dasar tenggorokan dekat dada; hembusan napas dalam.",
            'anatomy': "Pita suara di pangkal tenggorokan paling bawah terbuka agak lebar.",
            'tajweed_rule': 'Hams & Tarqiq (Desis Tipis)',
        },
        'خ': {
            'category': 'Adnal Halq (Ujung Tenggorokan)',
            'guidance': "Huruf Khaa (خ) keluar dari ujung tenggorokan paling atas dekat rongga mulut.",
            'anatomy': "Pertemuan ujung tenggorokan atas dengan pangkal langit-langit mulut.",
            'tajweed_rule': "Isti'la & Hams (Tebal & Desis)",
        },
        'غ': {
            'category': 'Adnal Halq (Ujung Tenggorokan)',
            'guidance': "Huruf Ghain (غ) keluar dari ujung tenggorokan; suara tebal bergema mengalir.",
            'anatomy': "Pangkal lidah paling belakang terangkat mendekati ujung tenggorokan.",
            'tajweed_rule': "Isti'la & Jahr (Tebal & Bergetar)",
        },
        'ص': {
            'category': "Isti'la / Tebal (Shafir)",
            'guidance': "Shaad (ص) huruf tebal & berdesis kuat. Bedakan dengan Siin (س) yang tipis.",
            'anatomy': "Ujung lidah di belakang gigi seri bawah, pangkal lidah terangkat membulat ke atas.",
            'tajweed_rule': 'Itbaq & Shafir (Tebal & Desis Tajam)',
        },
        'ض': {
            'category': 'Hafatul Lisan (Tepi Lidah)',
            'guidance': "Dhaad (ض) dari tepi lidah menempel ke geraham atas; jangan dibaca tipis seperti Daal (د).",
            'anatomy': "Salah satu atau kedua tepi lidah menempel kuat pada dinding gigi geraham atas.",
            'tajweed_rule': 'Istithalah (Suara Memanjang & Tebal)',
        },
        'ط': {
            'category': "Isti'la / Tebal (Itbaq)",
            'guidance': "Thaa (ط) huruf paling tebal; ujung lidah di pangkal gigi seri atas.",
            'anatomy': "Ujung lidah menempel di gusi gigi seri atas, seluruh badan lidah terangkat.",
            'tajweed_rule': "Itbaq & Qalqalah (Tebal & Memantul)",
        },
        'ظ': {
            'category': 'Tharful Lisan (Ujung Lidah)',
            'guidance': "Zhaa (ظ) dilafalkan tebal dengan ujung lidah sedikit keluar menyentuh ujung gigi seri atas.",
            'anatomy': "Ujung permukaan lidah sedikit menyentuh ujung dua gigi seri atas secara tebal.",
            'tajweed_rule': "Itbaq & Isti'la (Tebal Penuh)",
        },
        'ق': {
            'category': 'Aqshal Lisan (Pangkal Lidah)',
            'guidance': "Qaaf (ق) keluar dari pangkal lidah paling belakang menempel langit-langit lunak.",
            'anatomy': "Pangkal lidah paling belakang menempel rapat ke langit-langit lunak (uvula).",
            'tajweed_rule': "Isti'la & Qalqalah (Tebal & Memantul Kuat)",
        },
        'ك': {
            'category': 'Aqshal Lisan (Pangkal Lidah)',
            'guidance': "Kaf (ك) keluar dari pangkal lidah sedikit di depan posisi Qaaf (ق); tipis berhembus.",
            'anatomy': "Pangkal lidah bagian depan menempel ke langit-langit keras lalu terlepas berhembus.",
            'tajweed_rule': 'Hams & Tarqiq (Desis Tipis)',
        },
        'ث': {
            'category': 'Tharful Lisan (Ujung Lidah)',
            'guidance': "Tsaa (ث) dengan ujung lidah sedikit keluar menyentuh ujung gigi seri atas secara lembut.",
            'anatomy': "Ujung lidah dijepit halus di antara dua gigi seri depan.",
            'tajweed_rule': 'Hams & Rakhawah (Desis Soft)',
        },
        'ذ': {
            'category': 'Tharful Lisan (Ujung Lidah)',
            'guidance': "Dzal (ذ) tipis bergetar; ujung lidah di ujung dua gigi seri atas.",
            'anatomy': "Ujung lidah menyentuh bagian ujung gigi seri atas tanpa ditekan keras.",
            'tajweed_rule': 'Jahr & Tarqiq (Bergetar Tipis)',
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

        s = word.replace('\u0671', 'ا').replace('\u06E5', 'و').replace('\u06E6', 'ي')

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
            if harakat in {'\u064E', '\u064B'}:
                v = 'a'
            elif harakat in {'\u0650', '\u064D'}:
                v = 'i'
            elif harakat in {'\u064F', '\u064C'}:
                v = 'u'
            elif harakat == '\u0670':  # Dagger Alif (Superscript Alif Uthmani)
                v = 'aa'

            is_shaddah = len(letter) == 2
            base = letter[0] if is_shaddah else letter

            if base == 'ا':
                result.append('aa')
                continue
            if base == 'و':
                if v == 'u' or v == 'uu':
                    result.append('uu')
                    continue
                else:
                    result.append('w')
                    continue
            if base == 'ى' or base == 'ي':
                if v == 'i' or v == 'ii':
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
                if v == 'aa':
                    result.append('aa')
                elif v:
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
    def _word_similarity(cls, w1: str, w2: str) -> float:
        """Similarity [0.0, 1.0] antara 2 kata Arab berdasarkan Skeleton Normalized."""
        n1 = cls.normalize_arabic(w1)
        n2 = cls.normalize_arabic(w2)
        if not n1 or not n2:
            return 0.0
        if n1 == n2:
            return 1.0
        import difflib
        return difflib.SequenceMatcher(None, n1, n2).ratio()

    @classmethod
    def evaluate_realtime_stream(
        cls,
        target_ayah_text: str,
        recognized_speech_text: str
    ) -> Dict[str, Any]:
        """Evaluasi akurat real-time berbasis phoneme matching + word anchor alignment."""
        target_tokens = cls.arabic_to_phonemes(target_ayah_text)
        if not target_tokens:
            return cls._empty_result('Tidak ada target ayat.')

        if not recognized_speech_text:
            rec_tokens = []
        else:
            rec_tokens = cls.arabic_to_phonemes(recognized_speech_text)

        if not rec_tokens:
            return cls._empty_result('Belum ada bacaan yang terdeteksi.')

        ops = cls.aligned_diff(target_tokens, rec_tokens)
        matched = sum(1 for op, _, _ in ops if op == 'equal')
        total = len(target_tokens)
        rec_total = len(rec_tokens)
        partial_accuracy = round((matched / rec_total) * 100, 1) if rec_total else 0.0
        word_results, makhraj_errors = cls._analyze_word_level(
            target_ayah_text, target_tokens, rec_tokens, ops
        )

        target_words_list = [w for w in target_ayah_text.strip().split() if w]
        rec_words_list = [w for w in recognized_speech_text.strip().split() if w]
        matched_target_indices = set()

        for t_idx, t_w in enumerate(target_words_list):
            for r_w in rec_words_list:
                sim = cls._word_similarity(t_w, r_w)
                if sim >= 0.75:
                    matched_target_indices.add(t_idx)
                    break

        total_words_count = len(target_words_list)
        matched_words_count = sum(1 for w in word_results if w['status'] == 'matched')
        wrong_words_count = sum(1 for w in word_results if w['status'] == 'wrong')
        missed_words_count = total_words_count - matched_words_count - wrong_words_count

        if total_words_count > 0:
            if len(makhraj_errors) == 0 and matched_words_count == total_words_count:
                accuracy = 100
            else:
                # Akurasi berbasis persentase kata yang BENAR-BENAR FASIH (100% makhraj & harakat a/i/u benar)
                word_score = (matched_words_count / total_words_count) * 100
                # Penalti tegas Tajweed: -10% per kesalahan harakat/makhraj
                error_penalty = len(makhraj_errors) * 10
                accuracy = max(5, round(word_score - error_penalty))
        else:
            accuracy = 0

        passed = accuracy >= 85
        feedback = cls._generate_feedback(accuracy, passed, len(makhraj_errors))

        # Hitung Star Rating (1, 2, atau 3 bintang)
        if accuracy >= 90 and len(makhraj_errors) == 0:
            stars = 3
        elif accuracy >= 75:
            stars = 2
        else:
            stars = 1

        # Tentukan Kata Utama yang Perlu Di-Drill (jika ada kesalahan)
        primary_drill_word = ""
        if makhraj_errors:
            primary_drill_word = makhraj_errors[0].get('target_word', '')
        if not primary_drill_word:
            for w in word_results:
                if w.get('status') == 'wrong':
                    primary_drill_word = w.get('word', '')
                    break

        read_words_count = matched_words_count + wrong_words_count

        return {
            'accuracy': accuracy,
            'partial_accuracy': partial_accuracy,
            'rec_phoneme_count': rec_total,
            'passed': passed,
            'stars': stars,
            'primary_drill_word': primary_drill_word,
            'matched_count': matched_words_count,
            'read_count': read_words_count,
            'wrong_count': wrong_words_count,
            'missed_count': missed_words_count,
            'total_words': total_words_count,
            'total_phonemes': total,
            'matched_phonemes': matched,
            'word_results': word_results,
            'makhraj_errors': makhraj_errors,
            'teacher_feedback': feedback,
            'target_phonemes': ' '.join(target_tokens),
            'recognized_phonemes': ' '.join(rec_tokens),
            'recognized_speech_text': recognized_speech_text,
        }

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
                    target_w = words_arabic[word_idx]
                    
                    vowel_names = {
                        'a': 'Fathah (a)',
                        'i': 'Kasrah (i)',
                        'u': 'Dhommah (u)',
                        'aa': 'Mad Fathah (aa)',
                        'ii': 'Mad Kasrah (ii)',
                        'uu': 'Mad Dhommah (uu)',
                    }

                    if r_tok in {'a', 'i', 'u', 'aa', 'ii', 'uu'} and t_tok in {'a', 'i', 'u', 'aa', 'ii', 'uu'}:
                        t_name = vowel_names.get(t_tok, t_tok)
                        r_name = vowel_names.get(r_tok, r_tok)
                        if t_tok in {'aa', 'ii', 'uu'} or r_tok in {'aa', 'ii', 'uu'}:
                            word_errors[word_idx].append({
                                'type': 'mad',
                                'target': t_tok,
                                'detected': r_tok,
                                'target_char': t_name,
                                'detected_char': r_name,
                                'category': 'Kesalahan Mad (Durasi)',
                                'guidance': f"Pada kata '{target_w}', vokal {t_name} terucap sebagai {r_name}. Perhatikan durasi mad (2 harakat)."
                            })
                        else:
                            word_errors[word_idx].append({
                                'type': 'harakat',
                                'target_vowel': t_tok,
                                'detected_vowel': r_tok,
                                'target_char': t_name,
                                'detected_char': r_name,
                                'category': 'Kesalahan Harakat (Vokal)',
                                'guidance': f"Pada kata '{target_w}', vokal {t_name} terucap sebagai {r_name}. Pastikan dibaca dengan harakat {t_name} yang jelas."
                            })
                    else:
                        arabic_char = cls._PHONEME_TO_ARABIC.get(t_tok, t_tok)
                        detected_char = cls._PHONEME_TO_ARABIC.get(r_tok, r_tok)
                        info = cls.MAKHRAJ_GUIDANCE.get(arabic_char, {})
                        guidance = info.get('guidance', f"Pada kata '{target_w}', huruf '{arabic_char}' terucap mirip '{detected_char}'. Periksa makhraj.")
                        category = info.get('category', 'Artikulasi Makhraj')
                        anatomy = info.get('anatomy', 'Perhatikan posisi lidah dan rongga tenggorokan saat melafalkan huruf.')
                        tajweed_rule = info.get('tajweed_rule', 'Sifat & Makhraj Huruf')
                        word_errors[word_idx].append({
                            'type': 'makhraj',
                            'target_char': arabic_char,
                            'detected_char': detected_char,
                            'category': category,
                            'guidance': guidance,
                            'anatomy': anatomy,
                            'tajweed_rule': tajweed_rule,
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
            ratio = matched / total if total > 0 else 0
            has_error = len(word_errors.get(i, [])) > 0

            if total == 0 or ratio < 0.50:
                status = 'unread'
            elif has_error or ratio < 0.75:
                status = 'wrong'
            else:
                status = 'matched'
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
