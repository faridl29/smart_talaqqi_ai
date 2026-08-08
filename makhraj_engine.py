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
import difflib
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("MakhrajEngine")


class MakhrajEngine:
    """Engine Analisis Makhraj Akustik Real-Time berbasis PHONEME MATCHING."""

    # Pre-compiled regex patterns — menghindari re-compilation setiap panggilan
    _RE_HARAKAT = re.compile('[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]')
    _RE_INVISIBLE = re.compile('[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]')
    _RE_QURANIC = re.compile('[\u0610-\u061A\u06D6-\u06ED]')
    _RE_TASHKEEL = re.compile('[\u064B-\u065F\u0670]')
    _RE_SMALL = re.compile('[\u06E5\u06E6]')
    _RE_ALEF_VARIANTS = re.compile('[\u0625\u0623\u0622\u0671\u0672\u0673\u0675]')
    _RE_WHITESPACE = re.compile(r'\s+')

    MAKHRAJ_GUIDANCE = {
        'id': {
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
        },
        'en': {
            'ح': {
                'category': 'Wasathul Halq (Middle Throat)',
                'guidance': "The letter Haa (ح) comes from the middle of the throat with a clean sound.",
                'anatomy': "Press the epiglottis gently against the throat wall.",
                'tajweed_rule': 'Hams & Rakhawah (Whispered & Soft)',
            },
            'ع': {
                'category': 'Wasathul Halq (Middle Throat)',
                'guidance': "The letter 'Ain (ع) comes from the middle of the throat with slight pressure.",
                'anatomy': "Constriction in the middle throat with moderate vocal cord vibration.",
                'tajweed_rule': 'Tawassut (Moderate Sound)',
            },
            'ه': {
                'category': 'Aqshal Halq (Deep Throat)',
                'guidance': "The letter Ha (ه) originates from the bottom of the throat near the chest.",
                'anatomy': "Vocal cords at the lowest part of the throat open slightly wide.",
                'tajweed_rule': 'Hams & Tarqiq (Whispered & Thin)',
            },
            'خ': {
                'category': 'Adnal Halq (Upper Throat)',
                'guidance': "The letter Khaa (خ) comes from the upper throat near the mouth cavity.",
                'anatomy': "Contact between upper throat and soft palate root.",
                'tajweed_rule': "Isti'la & Hams (Heavy & Whispered)",
            },
            'غ': {
                'category': 'Adnal Halq (Upper Throat)',
                'guidance': "The letter Ghain (غ) comes from the upper throat with a heavy sound.",
                'anatomy': "Back of the tongue raised towards the upper throat.",
                'tajweed_rule': "Isti'la & Jahr (Heavy & Voiced)",
            },
            'ص': {
                'category': "Isti'la / Heavy (Shafir)",
                'guidance': "Shaad (ص) is a heavy letter with strong whistling sound.",
                'anatomy': "Tongue tip behind lower front teeth, back of tongue elevated.",
                'tajweed_rule': 'Itbaq & Shafir (Heavy & Sharp Whistle)',
            },
            'ض': {
                'category': 'Hafatul Lisan (Tongue Side)',
                'guidance': "Dhaad (ض) comes from the side of the tongue touching the upper molars.",
                'anatomy': "One or both sides of the tongue press firmly against upper molars.",
                'tajweed_rule': 'Istithalah (Elongated & Heavy)',
            },
            'ط': {
                'category': "Isti'la / Heavy (Itbaq)",
                'guidance': "Thaa (ط) is the heaviest letter; tongue tip at root of upper front teeth.",
                'anatomy': "Tongue tip at gums of upper front teeth, entire tongue elevated.",
                'tajweed_rule': "Itbaq & Qalqalah (Heavy & Echoing)",
            },
            'ظ': {
                'category': 'Tharful Lisan (Tongue Tip)',
                'guidance': "Zhaa (ظ) is pronounced heavy with tongue tip touching upper front teeth.",
                'anatomy': "Tip of tongue slightly touches edges of two upper front teeth.",
                'tajweed_rule': "Itbaq & Isti'la (Full Heavy)",
            },
            'ق': {
                'category': 'Aqshal Lisan (Back of Tongue)',
                'guidance': "Qaaf (ق) comes from the backmost part of the tongue against the soft palate.",
                'anatomy': "Back of tongue pressed tightly against soft palate.",
                'tajweed_rule': "Isti'la & Qalqalah (Heavy & Strong Echo)",
            },
            'ك': {
                'category': 'Aqshal Lisan (Back of Tongue)',
                'guidance': "Kaf (ك) comes slightly forward from Qaaf; thin and whispered.",
                'anatomy': "Front part of back tongue touches hard palate then releases breath.",
                'tajweed_rule': 'Hams & Tarqiq (Whispered & Thin)',
            },
            'ث': {
                'category': 'Tharful Lisan (Tongue Tip)',
                'guidance': "Tsaa (ث) with tongue tip slightly out, touching edge of upper front teeth.",
                'anatomy': "Tongue tip placed softly between upper and lower front teeth.",
                'tajweed_rule': 'Hams & Rakhawah (Soft Whispered)',
            },
        },
        'ar': {
            'ح': {
                'category': 'وسط الحلق',
                'guidance': "يخرج حرف الحاء (ح) من وسط الحلق بصوت صافٍ ولطيف.",
                'anatomy': "اضغط على وسط الحلق (لسان المزمار) برفق.",
                'tajweed_rule': 'الهمس والرخاوة',
            },
            'ع': {
                'category': 'وسط الحلق',
                'guidance': "يخرج حرف العين (ع) من وسط الحلق مع ضغط خفيف.",
                'anatomy': "تضييق في وسط الحلق مع اهتزاز الأوتار الصوتية.",
                'tajweed_rule': 'التوسط',
            },
            'ه': {
                'category': 'أقصى الحلق',
                'guidance': "يخرج حرف الهاء (هـ) من أقصى الحلق قريباً من الصدر.",
                'anatomy': "انفتاح الأوتار الصوتية في أسفل الحلق.",
                'tajweed_rule': 'الهمس والترقيق',
            },
            'خ': {
                'category': 'أدنى الحلق',
                'guidance': "يخرج حرف الخاء (خ) من أدنى الحلق قريباً من الفم.",
                'anatomy': "التقاء أدنى الحلق مع حنك الفم الرخو.",
                'tajweed_rule': 'الاستعلاء والهمس',
            },
            'غ': {
                'category': 'أدنى الحلق',
                'guidance': "يخرج حرف الغين (غ) من أدنى الحلق بصوت مفخم.",
                'anatomy': "ارتفاع أقصى اللسان نحو أدنى الحلق.",
                'tajweed_rule': 'الاستعلاء والجهر',
            },
            'ص': {
                'category': 'حرف مفخم (الصفير)',
                'guidance': "حرف الصاد (ص) مفخم وفيه صفير قوي.",
                'anatomy': "طرف اللسان خلف الأسنان السفلى مع ارتفاع أقصى اللسان.",
                'tajweed_rule': 'الإطباق والصفير',
            },
            'ض': {
                'category': 'حافة اللسان',
                'guidance': "يخرج حرف الضاد (ض) من إحدى حافتي اللسان مع الأضراس العليا.",
                'anatomy': "اتصال حافة اللسان بالأضراس العليا.",
                'tajweed_rule': 'الاستطالة والتفخيم',
            },
            'ط': {
                'category': 'حرف مفخم (الإطباق)',
                'guidance': "حرف الطاء (ط) أقوى الحروف تفخيماً.",
                'anatomy': "طرف اللسان عند أصول الثنايا العليا مع انطباق اللسان.",
                'tajweed_rule': 'الإطباق والقلقلة',
            },
            'ظ': {
                'category': 'طرف اللسان',
                'guidance': "يخرج حرف الظاء (ظ) مفخماً مع خروج طرف اللسان قليلاً.",
                'anatomy': "ملامسة طرف اللسان لأطراف الثنايا العليا.",
                'tajweed_rule': 'الإطباق والاستعلاء',
            },
            'ق': {
                'category': 'أقصى اللسان',
                'guidance': "يخرج حرف القاف (ق) من أقصى اللسان مع الحنك الأعلى.",
                'anatomy': "التصاق أقصى اللسان بالحنك الرخو.",
                'tajweed_rule': 'الاستعلاء والقلقلة',
            },
            'ك': {
                'category': 'أقصى اللسان',
                'guidance': "يخرج حرف الكاف (ك) أسفل من القاف قليلاً وهو مهموس ترقيق.",
                'anatomy': "ملامسة اللسان للحنك الصلب ثم انبعاث الهواء.",
                'tajweed_rule': 'الهمس والترقيق',
            },
            'ث': {
                'category': 'طرف اللسان',
                'guidance': "يخرج حرف الثاء (ث) بملامسة طرف اللسان لأطراف الثنايا العليا.",
                'anatomy': "وضع طرف اللسان بين الثنايا العليا والسفلى برفق.",
                'tajweed_rule': 'الهمس والرخاوة',
            },
        }
    }

    # Panduan kategori hukum tajwid tingkat lanjut (Mad Far'i & Ghunnah Musyaddadah)
    TAJWEED_RULE_GUIDANCE = {
        'id': {
            'ghunnah_musyaddadah': {
                'rule_name': 'Ghunnah Musyaddadah',
                'category': 'Ghunnah (Dengung)',
                'guidance': "Huruf nun (ن) atau mim (م) bertasydid wajib didengungkan dengan jelas selama ± 2 harakat (ghunnah musyaddadah).",
                'anatomy': "Tahan dengung hidung (khaisyum) selama 2 ketukan sambil mempertahankan bunyi huruf.",
            },
            'mad_wajib_muttashil': {
                'rule_name': 'Mad Wajib Muttashil',
                'category': 'Mad Far\'i (4-5 Harakat)',
                'guidance': "Mad bertemu hamzah dalam satu kata: wajib dibaca panjang 4-5 harakat.",
                'anatomy': "Panjangkan suara mad tanpa terputus hingga bertemu hamzah pada kata yang sama.",
            },
            'mad_jaiz_munfashil': {
                'rule_name': 'Mad Jaiz Munfashil',
                'category': 'Mad Far\'i (4-5 Harakat)',
                'guidance': "Mad bertemu hamzah pada kata berikutnya: boleh dibaca 2, 4, atau 5 harakat.",
                'anatomy': "Panjangkan suara mad di akhir kata sebelum hamzah pada awal kata berikutnya.",
            },
            'mad_lazim': {
                'rule_name': 'Mad Lazim',
                'category': 'Mad Far\'i (6 Harakat)',
                'guidance': "Mad bertemu sukun asli (bukan karena waqaf) dalam satu kata: wajib dibaca 6 harakat penuh.",
                'anatomy': "Tahan suara mad selama 6 ketukan dengan kuat dan konsisten.",
            },
            'mad_aridh_lissukun': {
                'rule_name': 'Mad Aridh Lis-Sukun',
                'category': 'Mad Far\'i (2-6 Harakat)',
                'guidance': "Mad bertemu sukun karena berhenti (waqaf) di akhir ayat: boleh dibaca 2, 4, atau 6 harakat.",
                'anatomy': "Panjangkan suara mad saat berhenti di akhir kata.",
            },
            'mad_lin': {
                'rule_name': 'Mad Lin (Mad Layyin)',
                'category': 'Mad Far\'i (2-6 Harakat)',
                'guidance': "Huruf waw (و) atau ya (ي) bersukun yang didahului fathah, lalu bertemu sukun karena waqaf: dibaca lunak 2, 4, atau 6 harakat.",
                'anatomy': "Panjangkan bunyi waw/ya secara lunak tanpa tekanan saat berhenti.",
            },
        },
        'en': {
            'ghunnah_musyaddadah': {
                'rule_name': 'Ghunnah Musyaddadah',
                'category': 'Ghunnah (Nasalization)',
                'guidance': "Nun (ن) or Mim (م) with shaddah must be nasalized clearly for 2 counts.",
                'anatomy': "Hold nasal sound through the nose for 2 beats while maintaining letter sound.",
            },
            'mad_wajib_muttashil': {
                'rule_name': 'Mad Wajib Muttashil',
                'category': 'Mad Far\'i (4-5 Counts)',
                'guidance': "Mad followed by hamzah in the same word: must be prolonged 4-5 counts.",
                'anatomy': "Prolong the mad sound continuously until reaching the hamzah in the same word.",
            },
            'mad_jaiz_munfashil': {
                'rule_name': 'Mad Jaiz Munfashil',
                'category': 'Mad Far\'i (4-5 Counts)',
                'guidance': "Mad followed by hamzah in the next word: may be prolonged 2, 4, or 5 counts.",
                'anatomy': "Prolong the mad sound at the end of word before hamzah at start of next word.",
            },
            'mad_lazim': {
                'rule_name': 'Mad Lazim',
                'category': 'Mad Far\'i (6 Counts)',
                'guidance': "Mad followed by original sukun in the same word: must be prolonged 6 full counts.",
                'anatomy': "Hold the mad sound for 6 full beats strongly and consistently.",
            },
            'mad_aridh_lissukun': {
                'rule_name': 'Mad Aridh Lis-Sukun',
                'category': 'Mad Far\'i (2-6 Counts)',
                'guidance': "Mad followed by sukun due to stopping (waqaf): may be prolonged 2, 4, or 6 counts.",
                'anatomy': "Prolong the mad sound when stopping at the end of a word.",
            },
            'mad_lin': {
                'rule_name': 'Mad Lin',
                'category': 'Mad Far\'i (2-6 Counts)',
                'guidance': "Waw or Ya with sukun preceded by fathah before waqaf sukun: pronounced soft 2-6 counts.",
                'anatomy': "Prolong the waw/ya sound softly without pressure when stopping.",
            },
        },
        'ar': {
            'ghunnah_musyaddadah': {
                'rule_name': 'غنة مشددة',
                'category': 'الغنة',
                'guidance': "النون أو الميم المشددة تجب فيها الغنة بمقدار حركتين.",
                'anatomy': "إخراج الصوت من الخيشوم بمقدار حركتين.",
            },
            'mad_wajib_muttashil': {
                'rule_name': 'مد واجب متصل',
                'category': 'مد فرعي (4-5 حرَكات)',
                'guidance': "مجيء الهمزة بعد حرف المد في كلمة واحدة: يمد 4-5 حركات.",
                'anatomy': "مد الصوت متصلاً حتى الهمزة في نفس الكلمة.",
            },
            'mad_jaiz_munfashil': {
                'rule_name': 'مد جائز منفصل',
                'category': 'مد فرعي (4-5 حرَكات)',
                'guidance': "مجيء الهمزة بعد حرف المد في كلمة أخرى: يمد 2 أو 4 أو 5 حركات.",
                'anatomy': "مد الصوت في آخر الكلمة قبل الهمزة.",
            },
            'mad_lazim': {
                'rule_name': 'مد لازم',
                'category': 'مد فرعي (6 حرَكات)',
                'guidance': "مجيء السكون الأصلي بعد حرف المد: يمد 6 حركات لزوماً.",
                'anatomy': "تمكين المد 6 حركات كاملة بقوة.",
            },
            'mad_aridh_lissukun': {
                'rule_name': 'مد عارض للسكون',
                'category': 'مد فرعي (2-6 حرَكات)',
                'guidance': "مجيء السكون بسبب الوقف بعد حرف المد: يمد 2 أو 4 أو 6 حركات.",
                'anatomy': "مد الصوت عند الوقف في آخر الكلمة.",
            },
            'mad_lin': {
                'rule_name': 'مد لين',
                'category': 'مد فرعي (2-6 حرَكات)',
                'guidance': "الواو أو الياء الساكنة المفتوح ما قبلها عند الوقف: يمد 2 أو 4 أو 6 حركات.",
                'anatomy': "مد صوت الواو/الياء بلين دون تكلف.",
            },
        }
    }

    @classmethod
    def get_makhraj_info(cls, char: str, lang: str = "id") -> Dict[str, str]:
        code = lang.lower() if lang else "id"
        if code not in cls.MAKHRAJ_GUIDANCE:
            code = "id"
        lang_dict = cls.MAKHRAJ_GUIDANCE.get(code, cls.MAKHRAJ_GUIDANCE["id"])
        return lang_dict.get(char, {})

    @classmethod
    def get_tajweed_info(cls, rule_key: str, lang: str = "id") -> Dict[str, str]:
        code = lang.lower() if lang else "id"
        if code not in cls.TAJWEED_RULE_GUIDANCE:
            code = "id"
        lang_dict = cls.TAJWEED_RULE_GUIDANCE.get(code, cls.TAJWEED_RULE_GUIDANCE["id"])
        return lang_dict.get(rule_key, {})

    @classmethod
    def _strip_harakat(cls, text: str) -> str:
        """Hapus seluruh harakat & tanda baca (kecuali huruf & shadda/tasydid)."""
        if not text:
            return ''
        s = cls._RE_HARAKAT.sub('', text)
        s = cls._RE_INVISIBLE.sub('', s)
        return s.replace('\u0651', '')

    @classmethod
    def _detect_tajweed_rules(cls, arabic_text: str, lang: str = "id") -> Dict[int, List[Dict[str, Any]]]:
        """
        Deteksi aturan tajwid tingkat lanjut pada setiap kata target:
        - Ghunnah Musyaddadah (نّ / مّ bertasydid)
        - Mad Wajib Muttashil / Jaiz Munfashil / Lazim / Aridh Lis-Sukun / Lin
        Mengembalikan {word_index: [list of rule dicts]}
        """
        if not arabic_text:
            return {}
        words = [w for w in arabic_text.strip().split() if w]
        rules_by_word: Dict[int, List[Dict[str, Any]]] = {}

        for idx, word in enumerate(words):
            rules: List[Dict[str, Any]] = []
            stripped = cls._strip_harakat(word)

            # ── 1. Ghunnah Musyaddadah: نّ / مّ (shadda bisa setelah harakat di Uthmani) ──
            for i, c in enumerate(word):
                if c == '\u0651':  # Shadda
                    # Scan backward melewati harakat untuk menemukan huruf yang bertasydid
                    j = i - 1
                    while j >= 0 and '\u064B' <= word[j] <= '\u0652':
                        j -= 1
                    if j >= 0 and word[j] in ('ن', 'م'):
                        info = cls.get_tajweed_info('ghunnah_musyaddadah', lang)
                        rules.append({
                            'type': 'ghunnah_musyaddadah',
                            'letter': word[j] + '\u0651',
                            'rule_name': info.get('rule_name', 'Ghunnah Musyaddadah'),
                            'category': info.get('category', 'Ghunnah'),
                            'guidance': info.get('guidance', ''),
                            'anatomy': info.get('anatomy', ''),
                        })

            # ── 2. Deteksi huruf mad & pengikutnya ──
            alif_positions = [i for i, c in enumerate(stripped) if c == 'ا']
            waw_ya_positions = [
                (i, c) for i, c in enumerate(stripped) if c in ('و', 'ي')
                and i + 1 < len(stripped)
                and stripped[i + 1] not in ('و', 'ي', 'ا')
            ]

            # 2a. Alif setelah fathah (mad thabi'i / mad far'i)
            for pos in alif_positions:
                if pos == 0:
                    continue
                nxt = stripped[pos + 1] if pos + 1 < len(stripped) else None
                # Mad Aridh Lis-Sukun: alif di akhir kata (berlaku saat waqaf)
                if nxt is None:
                    info = cls.get_tajweed_info('mad_aridh_lissukun', lang)
                    rules.append({
                        'type': 'mad_aridh_lissukun',
                        'letter': 'ا',
                        'rule_name': info.get('rule_name', 'Mad Aridh Lis-Sukun'),
                        'category': info.get('category', 'Mad Far\'i'),
                        'guidance': info.get('guidance', ''),
                        'anatomy': info.get('anatomy', ''),
                    })
                    continue
                # Mad Wajib Muttashil: alif bertemu hamzah pada kata yang sama
                if nxt == 'ء':
                    info = cls.get_tajweed_info('mad_wajib_muttashil', lang)
                    rules.append({
                        'type': 'mad_wajib_muttashil',
                        'letter': 'ا',
                        'rule_name': info.get('rule_name', 'Mad Wajib Muttashil'),
                        'category': info.get('category', 'Mad Far\'i'),
                        'guidance': info.get('guidance', ''),
                        'anatomy': info.get('anatomy', ''),
                    })
                    continue
                # Mad Lazim Kilmi: alif diikuti huruf bersukun asli dalam satu kata
                if nxt not in ('ا', 'و', 'ي', 'ء'):
                    info = cls.get_tajweed_info('mad_lazim', lang)
                    rules.append({
                        'type': 'mad_lazim',
                        'letter': 'ا',
                        'rule_name': info.get('rule_name', 'Mad Lazim'),
                        'category': info.get('category', 'Mad Far\'i'),
                        'guidance': info.get('guidance', ''),
                        'anatomy': info.get('anatomy', ''),
                    })
                    continue
                # Mad Jaiz Munfashil: alif di akhir kata sebelum hamzah (kata berikutnya)
                if idx + 1 < len(words):
                    next_word = cls._strip_harakat(words[idx + 1])
                    if next_word.startswith('ا') and len(next_word) > 1 and next_word[1] == 'ء':
                        info = cls.get_tajweed_info('mad_jaiz_munfashil', lang)
                        rules.append({
                            'type': 'mad_jaiz_munfashil',
                            'letter': 'ا',
                            'rule_name': info.get('rule_name', 'Mad Jaiz Munfashil'),
                            'category': info.get('category', 'Mad Far\'i'),
                            'guidance': info.get('guidance', ''),
                            'anatomy': info.get('anatomy', ''),
                        })

            # 2b. Waw / Ya: hanya Mad Lin di akhir kata atau Mad Wajib Muttashil dgn hamzah
            for pos, c in waw_ya_positions:
                # Mad Lin: waw/ya di akhir kata (berlaku saat waqaf)
                if pos == len(stripped) - 1:
                    info = cls.get_tajweed_info('mad_lin', lang)
                    rules.append({
                        'type': 'mad_lin',
                        'letter': c,
                        'rule_name': info.get('rule_name', 'Mad Lin'),
                        'category': info.get('category', 'Mad Far\'i'),
                        'guidance': info.get('guidance', ''),
                        'anatomy': info.get('anatomy', ''),
                    })
                    continue
                # Mad Wajib Muttashil: waw/ya bertemu hamzah dalam satu kata
                if stripped[pos + 1] == 'ء':
                    info = cls.get_tajweed_info('mad_wajib_muttashil', lang)
                    rules.append({
                        'type': 'mad_wajib_muttashil',
                        'letter': c,
                        'rule_name': info.get('rule_name', 'Mad Wajib Muttashil'),
                        'category': info.get('category', 'Mad Far\'i'),
                        'guidance': info.get('guidance', ''),
                        'anatomy': info.get('anatomy', ''),
                    })

            # Hindari duplikasi rule yang sama pada satu kata
            seen: set = set()
            unique_rules = []
            for r in rules:
                key = r['type']
                if key not in seen:
                    seen.add(key)
                    unique_rules.append(r)
            rules_by_word[idx] = unique_rules

        return rules_by_word

    @classmethod
    def normalize_arabic(cls, text: str) -> str:
        """Normalisasi teks Arab murni (hapus harakat, kontrol, dll)."""
        if not text:
            return ''
        cleaned = cls._RE_INVISIBLE.sub('', text)
        cleaned = cls._RE_QURANIC.sub('', cleaned)
        cleaned = cls._RE_TASHKEEL.sub('', cleaned)
        cleaned = cleaned.replace('\u0640', '')
        cleaned = cls._RE_SMALL.sub('', cleaned)
        cleaned = cls._RE_ALEF_VARIANTS.sub('ا', cleaned).replace('\u0671', 'ا')
        cleaned = cleaned.replace('ة', 'ه').replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي').replace('ء', '')
        cleaned = cls._RE_WHITESPACE.sub(' ', cleaned).strip()
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
        'و': 'w',
        'ي': 'y',
        'ى': 'y',
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
                    # Shadda mark — double huruf terakhir
                    if chars and chars[-1][0]:
                        letter = chars[-1][0]
                        harakat_val = chars[-1][1]
                        # Langsung double huruf (baik sudah punya harakat atau belum)
                        chars[-1] = (letter + letter, harakat_val)
                    else:
                        pending_shadda = True
                elif chars and chars[-1][1] is None:
                    # Harakat biasa → assign ke huruf terakhir
                    if pending_shadda:
                        # Shadda + Harakat pada huruf yang sama (urutan: ب + ّ + ِ)
                        chars[-1] = (chars[-1][0] + chars[-1][0], c)
                        pending_shadda = False
                    else:
                        chars[-1] = (chars[-1][0], c)
                else:
                    chars.append(('', c))
                j += 1
                continue
            elif '\u0600' <= c <= '\u06FF':
                if pending_shadda and chars and chars[-1][0]:
                    # Shadda tanpa harakat sebelum huruf baru — double dulu
                    chars[-1] = (chars[-1][0] + chars[-1][0], chars[-1][1])
                    pending_shadda = False
                chars.append((c, None))
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
                result.append(v if v else 'a')
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
        if len(n1) <= 3 or len(n2) <= 3:
            return 1.0 if n1 == n2 else 0.0
        return difflib.SequenceMatcher(None, n1, n2).ratio()

    @classmethod
    def evaluate_realtime_stream(
        cls,
        target_ayah_text: str,
        recognized_speech_text: str,
        lang: str = "id"
    ) -> Dict[str, Any]:
        """Evaluasi akurat real-time berbasis phoneme matching + word anchor alignment."""
        target_tokens = cls.arabic_to_phonemes(target_ayah_text)
        if not target_tokens:
            return cls._empty_result('no_target', lang=lang)

        if not recognized_speech_text:
            rec_tokens = []
        else:
            rec_tokens = cls.arabic_to_phonemes(recognized_speech_text)

        if not rec_tokens:
            return cls._empty_result('no_speech', lang=lang)

        ops = cls.aligned_diff(target_tokens, rec_tokens)
        matched = sum(1 for op, _, _ in ops if op == 'equal')
        total = len(target_tokens)
        rec_total = len(rec_tokens)
        partial_accuracy = round((matched / rec_total) * 100, 1) if rec_total else 0.0
        word_results, makhraj_errors = cls._analyze_word_level(
            target_ayah_text, target_tokens, rec_tokens, ops, lang=lang
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
                # Akurasi Tajweed Presisi: Kata 100% fasih = 1.0, kata salah (wrong) hanya dapat partial credit 0.1
                raw_score = ((matched_words_count + 0.1 * wrong_words_count) / total_words_count) * 100
                accuracy = max(5, min(99, round(raw_score)))
        else:
            accuracy = 0

        passed = accuracy >= 85
        feedback = cls._generate_feedback(accuracy, passed, len(makhraj_errors), lang=lang)

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

        # Deteksi aturan tajwid tingkat lanjut per-kata (Ghunnah Musyaddadah & Mad Far'i)
        tajweed_rules = cls._detect_tajweed_rules(target_ayah_text, lang=lang)

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
            'tajweed_rules': tajweed_rules,
            'teacher_feedback': feedback,
            'target_phonemes': ' '.join(target_tokens),
            'recognized_phonemes': ' '.join(rec_tokens),
            'recognized_speech_text': recognized_speech_text,
            'diagnosis_basis': 'phoneme_text_matching',
        }

    @classmethod
    def _analyze_word_level(
        cls,
        original_text: str,
        target_tokens: List[str],
        rec_tokens: List[str],
        ops: List[Tuple[str, str, str]],
        lang: str = "id"
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

        code = (lang or "id").lower()

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
                    
                    is_last_word = (word_idx == len(words_arabic) - 1)
                    is_end_vowel_at_waqf = is_last_word and (target_idx >= len(target_token_to_word) - 2) and (t_tok in {'a', 'i', 'u'})

                    if is_end_vowel_at_waqf:
                        # Hukum Tajweed Waqf Bil Iskan: Vokal pendek akhir di ujung ayat sah dibaca Sukun / berbeda
                        word_matched[word_idx] = word_matched.get(word_idx, 0) + 1
                    else:
                        if code == 'en':
                            vowel_names = {
                                'a': 'Fathah (a)', 'i': 'Kasrah (i)', 'u': 'Dhommah (u)',
                                'aa': 'Mad Fathah (aa)', 'ii': 'Mad Kasrah (ii)', 'uu': 'Mad Dhommah (uu)',
                            }
                        elif code == 'ar':
                            vowel_names = {
                                'a': 'فتحة (a)', 'i': 'كسرة (i)', 'u': 'ضمة (u)',
                                'aa': 'مد بالفتح (aa)', 'ii': 'مد بالكسر (ii)', 'uu': 'مد بالضم (uu)',
                            }
                        else:
                            vowel_names = {
                                'a': 'Fathah (a)', 'i': 'Kasrah (i)', 'u': 'Dhommah (u)',
                                'aa': 'Mad Fathah (aa)', 'ii': 'Mad Kasrah (ii)', 'uu': 'Mad Dhommah (uu)',
                            }

                        if r_tok in {'a', 'i', 'u', 'aa', 'ii', 'uu'} and t_tok in {'a', 'i', 'u', 'aa', 'ii', 'uu'}:
                            t_name = vowel_names.get(t_tok, t_tok)
                            r_name = vowel_names.get(r_tok, r_tok)
                            if t_tok in {'aa', 'ii', 'uu'} or r_tok in {'aa', 'ii', 'uu'}:
                                if code == 'en':
                                    cat = 'Mad Duration Error'
                                    guid = f"In word '{target_w}', vowel {t_name} was pronounced as {r_name}. Mind the mad duration (2 counts)."
                                elif code == 'ar':
                                    cat = 'خطأ في المد'
                                    guid = f"في كلمة '{target_w}'، نُطقت حركة {t_name} كـ {r_name}. انتبه لمقدار المد (حركتان)."
                                else:
                                    cat = 'Kesalahan Mad (Durasi)'
                                    guid = f"Pada kata '{target_w}', vokal {t_name} terucap sebagai {r_name}. Perhatikan durasi mad (2 harakat)."

                                word_errors[word_idx].append({
                                    'type': 'mad',
                                    'target': t_tok,
                                    'detected': r_tok,
                                    'target_char': t_name,
                                    'detected_char': r_name,
                                    'category': cat,
                                    'guidance': guid
                                })
                            else:
                                if code == 'en':
                                    cat = 'Vowel Harakat Error'
                                    guid = f"In word '{target_w}', vowel {t_name} was pronounced as {r_name}. Ensure it is recited with clear {t_name}."
                                elif code == 'ar':
                                    cat = 'خطأ في الحركات'
                                    guid = f"في كلمة '{target_w}'، نُطقت الحركة {t_name} كـ {r_name}. ينبغي بيان حركة {t_name} بوضوح."
                                else:
                                    cat = 'Kesalahan Harakat (Vokal)'
                                    guid = f"Pada kata '{target_w}', vokal {t_name} terucap sebagai {r_name}. Pastikan dibaca dengan harakat {t_name} yang jelas."

                                word_errors[word_idx].append({
                                    'type': 'harakat',
                                    'target_vowel': t_tok,
                                    'detected_vowel': r_tok,
                                    'target_char': t_name,
                                    'detected_char': r_name,
                                    'category': cat,
                                    'guidance': guid
                                })
                        else:
                            arabic_char = cls._PHONEME_TO_ARABIC.get(t_tok, t_tok)
                            detected_char = cls._PHONEME_TO_ARABIC.get(r_tok, r_tok)
                            info = cls.get_makhraj_info(arabic_char, code)

                            if code == 'en':
                                fallback_guid = f"In word '{target_w}', letter '{arabic_char}' sounded like '{detected_char}'. Check articulation point."
                                fallback_cat = 'Makhraj Articulation'
                                fallback_anat = 'Check tongue position and throat cavity.'
                                fallback_rule = 'Letter Makhraj & Characteristics'
                            elif code == 'ar':
                                fallback_guid = f"في كلمة '{target_w}'، نُطق حرف '{arabic_char}' قريباً من '{detected_char}'. يرجى تحري المخرج."
                                fallback_cat = 'مخرج الحرف'
                                fallback_anat = 'انتبه لموضع اللسان والتجويف الحلقي.'
                                fallback_rule = 'مخرج وصفات الحرف'
                            else:
                                fallback_guid = f"Pada kata '{target_w}', huruf '{arabic_char}' terucap mirip '{detected_char}'. Periksa makhraj."
                                fallback_cat = 'Artikulasi Makhraj'
                                fallback_anat = 'Perhatikan posisi lidah dan rongga tenggorokan saat melafalkan huruf.'
                                fallback_rule = 'Sifat & Makhraj Huruf'

                            guidance = info.get('guidance', fallback_guid)
                            category = info.get('category', fallback_cat)
                            anatomy = info.get('anatomy', fallback_anat)
                            tajweed_rule = info.get('tajweed_rule', fallback_rule)
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
                word_idx = target_token_to_word[target_idx] if target_idx < len(target_token_to_word) else (target_token_to_word[-1] if target_token_to_word else -1)
                if word_idx >= 0:
                    word_total[word_idx] = word_total.get(word_idx, 0) + 1
                    extra_char = cls._PHONEME_TO_ARABIC.get(r_tok, r_tok)
                    if r_tok not in {'a', 'i', 'u'}:
                        if code == 'en':
                            cat = 'Extra Letter/Vowel'
                            guid = f"Extra sound '{extra_char}' detected in word '{words_arabic[word_idx]}'."
                        elif code == 'ar':
                            cat = 'زيادة حرف أو حركة'
                            guid = f"تم رصد زيادة حرف/حركة '{extra_char}' في كلمة '{words_arabic[word_idx]}'."
                        else:
                            cat = 'Penambahan Huruf/Vokal'
                            guid = f"Terdapat penambahan sebutan/vokal '{extra_char}' pada kata '{words_arabic[word_idx]}'."

                        word_errors[word_idx].append({
                            'type': 'makhraj_extra',
                            'target_char': '',
                            'detected_char': extra_char,
                            'category': cat,
                            'guidance': guid,
                        })
                rec_idx += 1
            elif op == 'del':
                word_idx = target_token_to_word[target_idx] if target_idx < len(target_token_to_word) else -1
                if word_idx >= 0:
                    word_total[word_idx] = word_total.get(word_idx, 0) + 1
                    is_last_word = (word_idx == len(words_arabic) - 1)
                    is_end_vowel_at_waqf = is_last_word and (target_idx >= len(target_token_to_word) - 2) and (t_tok in {'a', 'i', 'u'})

                    if is_end_vowel_at_waqf:
                        # Hukum Tajweed Waqf Bil Iskan: Hilang vokal pendek di akhir ayat dianggap matched
                        word_matched[word_idx] = word_matched.get(word_idx, 0) + 1
                    elif t_tok not in {'a', 'i', 'u', 'aa', 'ii', 'uu'}:
                        arabic_char = cls._PHONEME_TO_ARABIC.get(t_tok, t_tok)
                        if code == 'en':
                            cat = 'Missing Letter'
                            guid = f"Letter '{arabic_char}' was not heard. Ensure it is recited completely."
                        elif code == 'ar':
                            cat = 'حرف مفقود'
                            guid = f"لم يُسمع حرف '{arabic_char}'. يرجى إتمامه في القراءة."
                        else:
                            cat = 'Huruf Tidak Terdeteksi'
                            guid = f"Huruf '{arabic_char}' tidak terdengar. Pastikan dibaca dengan lengkap."

                        word_errors[word_idx].append({
                            'type': 'makhraj_missing',
                            'target_char': arabic_char,
                            'detected_char': '',
                            'category': cat,
                            'guidance': guid,
                        })
                target_idx += 1

        word_results = []
        for i, arabic_word in enumerate(words_arabic):
            matched = word_matched.get(i, 0)
            total = word_total.get(i, 0)
            ratio = matched / total if total > 0 else 0
            errs = word_errors.get(i, [])

            has_error = len(errs) > 0

            if total == 0 or matched == 0 or ratio < 0.35:
                status = 'unread'
            elif has_error or ratio < 0.90:
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
            w_status = word_results[w_idx]['status'] if w_idx < len(word_results) else 'unread'
            # Emit error detail untuk kata yang salah atau sedang di-evaluasi
            if w_status == 'unread':
                continue

            for err in errs:
                err_with_word = dict(err)
                err_with_word['target_word'] = words_arabic[w_idx]
                makhraj_errors.append(err_with_word)

        return word_results, makhraj_errors

    @classmethod
    def _empty_result(cls, msg_key: str, lang: str = "id") -> Dict[str, Any]:
        code = (lang or "id").lower()
        if msg_key == 'no_target':
            msg = "Tidak ada target ayat." if code == 'id' else ("No target ayah." if code == 'en' else "لا توجد آية محددة.")
        elif msg_key == 'no_speech':
            msg = "Belum ada bacaan yang terdeteksi." if code == 'id' else ("No recitation detected yet." if code == 'en' else "لم يتم رصد أي قراءة بعد.")
        else:
            msg = msg_key

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
    def _generate_feedback(accuracy: int, passed: bool, error_count: int, lang: str = "id") -> str:
        code = (lang or "id").lower()
        if code == "en":
            if passed:
                if accuracy == 100:
                    return "Masha Allah, perfect recitation!"
                return f"Good job ({accuracy}%). Mind {error_count} minor detail(s)."
            if accuracy == 0:
                return "No recitation detected."
            if accuracy < 50:
                return f"Try again more carefully ({accuracy}%)."
            return f"Almost there ({accuracy}%). Correct {error_count} error(s)."
        elif code == "ar":
            if passed:
                if accuracy == 100:
                    return "ما شاء الله، قراءة ممتازة ومتقنة!"
                return f"جيد جداً ({accuracy}%). انتبه لـ {error_count} ملاحظة."
            if accuracy == 0:
                return "لم يتم اكتشاف أي قراءة."
            if accuracy < 50:
                return f"حاول مرة أخرى بتركيز أكبر ({accuracy}%)."
            return f"قريب جداً ({accuracy}%). يرجى تصحيح {error_count} خطأ."
        else:
            if passed:
                if accuracy == 100:
                    return "Masya Allah, bacaan sempurna!"
                return f"Bagus ({accuracy}%). Perhatikan {error_count} detail kecil."
            if accuracy == 0:
                return "Tidak ada bacaan yang terdeteksi."
            if accuracy < 50:
                return f"Coba lagi dengan lebih teliti ({accuracy}%)."
            return f"Hampir ({accuracy}%). Perbaiki {error_count} kesalahan."

