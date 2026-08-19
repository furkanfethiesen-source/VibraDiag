

"""
VibraDiag - JSON Kaynakli Kural Motoru (Faz 5)
==================================================

Modul icerigi
-------------
1) fig_p15_0 -> ISO 2372 / VDI 2056 Severity Tablosu   (Adim 5.2)
2) table_p29_vision -> Ampirik Zarf Filtresi Tablosu   (Faz 4'un JSON-tabanli hali)
3) table_p47_0+p48 -> Pistonlu Makine Ariza Tablosu    (reciprocating_faults.py'nin JSON-tabanli hali)
4) visual_chunks.json -> Spektral Desen Capraz Kontrolu (Adim 5.3)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_DEFAULT_TABLE_CHUNKS_PATH = "data/processed/table_chunks.json"
_DEFAULT_VISUAL_CHUNKS_PATH = "data/processed/visual_chunks.json"

DEFAULT_TABLE_CHUNKS_PATH = _DEFAULT_TABLE_CHUNKS_PATH
DEFAULT_VISUAL_CHUNKS_PATH = _DEFAULT_VISUAL_CHUNKS_PATH


def load_chunks(json_path: str = _DEFAULT_TABLE_CHUNKS_PATH) -> list[dict]:
    """table_chunks.json / visual_chunks.json gibi chunk-listesi JSON
    dosyalarini yukler (notebook'taki load_tables() ile ayni islev,
    genellestirilmis isim)."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' bulunamadi. table_chunks.json/visual_chunks.json "
            "genelde data/processed/ altinda olur; farkli bir konumdaysa "
            "json_path parametresiyle belirt."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_chunk(chunks: list[dict], chunk_id: str) -> dict:
    """chunks listesinde chunk_id'ye tam eslesen kaydi bulur."""
    chunk = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
    if chunk is None:
        raise ValueError(f"'{chunk_id}' bulunamadi — chunk_id degismis olabilir")
    return chunk


_ROMAN_TO_ARABIC = {"I": 1, "II": 2, "III": 3, "IV": 4}
_ARABIC_TO_ROMAN = {v: k for k, v in _ROMAN_TO_ARABIC.items()}


def load_severity_table(chunks: list[dict]) -> dict:
    """table_chunks.json icinden fig_p15_0'i bulur ve severity lookup
    icin normalize edilmis bir dict dondurur (min/max None -> 0.0/inf)."""
    chunk = find_chunk(chunks, "fig_p15_0")
    raw = chunk["structured_data"]

    groups_by_name = {}
    for g in raw["groups"]:
        zones_clean = {}
        for zone_name, bounds in g["zones"].items():
            low = bounds["min_mm_s"] if bounds["min_mm_s"] is not None else 0.0
            high = bounds["max_mm_s"] if bounds["max_mm_s"] is not None else float("inf")
            zones_clean[zone_name] = {"min_mm_s": low, "max_mm_s": high}

        groups_by_name[g["name"]] = {
            "machine_description": g["machine_description"],
            "foundation_type": g["foundation_type"],
            "zones": zones_clean,
        }

    return {
        "standard": raw["standard"],
        "unit": raw.get("measurement_unit", "mm/s RMS"),
        "zone_meanings": raw["zone_meanings"],
        "groups": groups_by_name,
    }


def normalize_machine_class(user_input) -> str:
    """Kullanicinin 'Class II', 'class 2', '2', 'II' gibi cesitli
    girdilerini kanonik 'Class I'/'Class II'/... formatina cevirir."""
    if isinstance(user_input, int):
        arabic = user_input
    else:
        s = str(user_input).strip().upper().replace("_", " ").replace("-", " ").replace("CLASS", "").strip()
        if s.isdigit():
            arabic = int(s)
        elif s in _ROMAN_TO_ARABIC:
            return f"Class {s}"
        else:
            raise ValueError(f"Taninmayan makine sinifi: {user_input!r}")

    roman = _ARABIC_TO_ROMAN.get(arabic)
    if roman is None:
        raise ValueError(f"Gecersiz sinif numarasi: {arabic} (1-4 arasi olmali)")
    return f"Class {roman}"


def classify_severity(machine_class, rms_mm_s: float, severity_table: dict) -> dict:
    """RMS hiz degerini (mm/s) ISO 2372/VDI 2056 Zone A/B/C/D'ye cevirir.

    Notebook'taki versiyona gore fark: rms_mm_s < 0 icin ayri ve acik bir
    hata mesaji eklendi (eskiden dongu sonuna dusup belirsiz bir hata
    veriyordu)."""
    if rms_mm_s < 0:
        raise ValueError(f"rms_mm_s negatif olamaz: {rms_mm_s}")

    mc = normalize_machine_class(machine_class)
    group = severity_table["groups"].get(mc)
    if group is None:
        raise ValueError(f"Bilinmeyen makine sinifi: {machine_class}")

    for zone_name, bounds in group["zones"].items():
        if bounds["min_mm_s"] <= rms_mm_s < bounds["max_mm_s"]:
            return {
                "zone": zone_name,
                "meaning": severity_table["zone_meanings"][zone_name],
                "range_mm_s": (bounds["min_mm_s"], bounds["max_mm_s"]),
                "machine_class": mc,
                "machine_description": group["machine_description"],
            }

    raise ValueError(f"Zon bulunamadi (rms_mm_s={rms_mm_s}, sinif={mc})")



def parse_range(range_str: str) -> tuple[float, float]:
    """'5 – 100 Hz' / '2,500 – ... RPM' gibi aralik string'lerini
    (alt, ust) float ciftine cevirir. '...'/'…' -> sonsuz."""
    cleaned = range_str.replace(",", "").strip()
    parts = re.split(r"\s*[–-]\s*", cleaned)
    if len(parts) != 2:
        raise ValueError(f"Beklenmeyen aralik formati: {range_str!r}")

    def _to_float(token: str) -> float:
        token = re.sub(r"[A-Za-z]+", "", token).strip()
        if token in ("...", "…", ""):
            return float("inf")
        return float(token)

    return _to_float(parts[0]), _to_float(parts[1])


def load_envelope_filter_table(chunks: list[dict]) -> list[dict]:
    """table_p29_vision'i (Table 3.2 Typical Filter Setting for Envelope
    Analysis) yapilandirilmis bir listeye cevirir. empirical_band_rules.
    select_band_empirical() artik bu fonksiyonu kullaniyor — sabit
    _TABLE_P29_FILTERS sozlugu kaldirildi."""
    chunk = find_chunk(chunks, "table_p29_vision")
    filters = []
    for filter_no, freq_band, speed_range, analyzing_range in chunk["rows"]:
        filters.append({
            "filter_no": int(filter_no),
            "freq_band_hz": parse_range(freq_band),
            "speed_range_rpm": parse_range(speed_range),
            "analyzing_range_hz": parse_range(analyzing_range),
        })
    return filters


def clean_text(s: str) -> str:
    """Coklu-satirli tablo hucrelerini tek satira indirger, × -> x
    normalize eder."""
    s = s.replace("\n", " ").replace("×", "x")
    return re.sub(r"\s+", " ", s).strip()


_NUM_X_RPM = re.compile(r"(\d+(?:\.\d+)?)\s*x\s*(?:RPM)?", re.IGNORECASE)


def extract_all_multipliers(freq_str: str) -> list[float]:
    """Metindeki tum 'Nx' bicimli SAYISAL katsayilari yakalar (RPM
    kelimesi hemen yaninda olsun olmasin). Ornekler:
    'are 1x and 2xRPM' -> [1.0, 2.0]
    '2 x RPM and multiples' -> [2.0]
    """
    return [float(m) for m in _NUM_X_RPM.findall(clean_text(freq_str))]


_STROKE_TOKEN_PATTERN = re.compile(
    r"([\d.]+|N)\s*x\s*RPM\s*for\s*(4-stroke|2-stroke)", re.IGNORECASE
)


def parse_stroke_dependent_multipliers(freq_str: str) -> dict:
    """'0.5 x RPM for 4-stroke engines 1 x RPM for 2-stroke engines' gibi
    stroke-tipine bagli metinleri {"4-stroke": 0.5, "2-stroke": 1.0}
    seklinde ayristirir. 'N' (silindir sayisina bagli sembolik katsayi)
    string olarak dondurulur, sayisal degil."""
    cleaned = clean_text(freq_str)
    result: dict[str, float | str] = {}
    for token, stroke in _STROKE_TOKEN_PATTERN.findall(cleaned):
        result[stroke] = "N" if token.upper() == "N" else float(token)
    return result


_KNOWN_PARSE_OVERRIDES: dict[str, dict[str, float | str]] = {
    "Power pulses": {"4-stroke": "N", "2-stroke": "N/2"},
}


def parse_reciprocating_table(chunks: list[dict]) -> list[dict]:
    """table_p47_0+p48'i yapilandirilmis satir listesine cevirir. Her
    satir icin: description, multipliers (stroke-tipine gore ya da
    'any'/'reference'), cause, remedy, ve nasil parse edildigi (parse_note)
    doner — boylece hangi satirlarin otomatik, hangisinin manuel
    dogrulandigi izlenebilir kalir."""
    chunk = find_chunk(chunks, "table_p47_0+p48")

    rows = []
    for raw_description, predominant_freqs, cause, remedy in chunk["rows"]:
        description = clean_text(raw_description)

        override_key = next((k for k in _KNOWN_PARSE_OVERRIDES if k in description), None)
        if override_key:
            multipliers = dict(_KNOWN_PARSE_OVERRIDES[override_key])
            parse_note = "manuel override (PDF/OCR N/2 kesrini bozmus)"
        elif "same as" in predominant_freqs.lower():
            multipliers = {"reference": clean_text(predominant_freqs)}
            parse_note = "capraz referans (baska satira esit)"
        else:
            stroke_specific = parse_stroke_dependent_multipliers(predominant_freqs)
            if stroke_specific:
                multipliers = stroke_specific
                parse_note = "regex ile otomatik ayristirildi (stroke-bagimli)"
            else:
                values = extract_all_multipliers(predominant_freqs)
                multipliers = {"any": values} if values else {}
                parse_note = "regex ile otomatik ayristirildi (stroke-bagimsiz)"

        rows.append({
            "description": description,
            "raw_predominant_freqs": predominant_freqs,
            "multipliers": multipliers,
            "cause": clean_text(cause),
            "remedy": clean_text(remedy),
            "parse_note": parse_note,
        })

    return rows


_RELEVANT_FAULT_TYPES = {
    "unbalance", "angular_misalignment", "parallel_misalignment",
    "mechanical_looseness", "mechanical_looseness_internal",
}

_AMPLITUDE_RANK = {"dominant": 3, "moderate": 2, "low": 1}


def load_spectral_patterns(chunks: list[dict]) -> list[dict]:
    """visual_chunks.json'dan unbalance/misalignment/looseness ailesine ait
    (yani signal_processing.match_ratio_pattern'in kapsadigi uc kategoriyle
    ilgili) spektral desen kurallarini cikarir."""
    patterns = []
    for c in chunks:
        sd = c.get("structured_data", {})
        fault_type = sd.get("fault_type")
        if fault_type in _RELEVANT_FAULT_TYPES and sd.get("peaks"):
            patterns.append({
                "chunk_id": c["chunk_id"],
                "fault_type": fault_type,
                "direction": sd.get("direction"),
                "peaks": sd["peaks"],
                "distinguishing_rule": sd.get("distinguishing_rule"),
                "sideband_pattern": sd.get("sideband_pattern"),
                "confidence": c.get("confidence"),
            })
    return patterns


def _dominant_order_from_pattern(pattern: dict) -> str | None:
    peaks = pattern["peaks"]
    if not peaks:
        return None
    best = max(peaks, key=lambda p: _AMPLITUDE_RANK.get(p["relative_amplitude"], 0))
    return best["order"].upper()


def _dominant_order_from_amplitudes(harmonic_amplitudes: dict) -> str | None:
    if not harmonic_amplitudes:
        return None
    best_h = max(harmonic_amplitudes, key=harmonic_amplitudes.get)
    return f"{best_h}X"


def cross_check_ratio_pattern(ratio_results: dict, spectral_patterns: list[dict]) -> dict:
    """signal_processing.match_ratio_pattern() ciktisindaki sayisal
    harmonic_amplitudes'i, visual_chunks.json'daki nitel (dominant/
    moderate/low) desenlerle kiyaslar.

    ONEMLI SINIRLAMA: bu KESIN TANI degil, DESTEKLEYICI bir ipucudur.
    - Kiyaslama sadece "hangi harmonik baskin" sorusuna dayanir; kaynak
    desenlerin cogu ayrica bir YON (radyal/aksiyal) sarti da tasir
    (orn. angular_misalignment aksiyal yonde baskindir), ama bizim
    dogrudan-spektrum hattimiz (match_ratio_pattern) tek kanalli ve
    yon bilgisi tasimiyor — bu yuzden birden fazla aday ariza tipi
    ayni anda eslesebilir (orn. 2X baskinsa hem parallel_misalignment
    hem mechanical_looseness aday olur).
    - Kesin ayrim icin fault_localization.check_coupling_phase() (Faz 2)
    ile faz bilgisi veya coklu-yonlu (radyal+aksiyal) olcum eklenmeli.
    """
    observed_dominant = _dominant_order_from_amplitudes(ratio_results.get("harmonic_amplitudes", {}))

    matches = []
    for pattern in spectral_patterns:
        expected_dominant = _dominant_order_from_pattern(pattern)
        if expected_dominant and observed_dominant and expected_dominant == observed_dominant:
            matches.append({
                "fault_type": pattern["fault_type"],
                "chunk_id": pattern["chunk_id"],
                "expected_dominant_order": expected_dominant,
                "direction_required": pattern["direction"],
                "distinguishing_rule": pattern["distinguishing_rule"],
            })

    return {
        "observed_dominant_order": observed_dominant,
        "candidate_matches": matches,
        "note": (
            "Bu kiyaslama sadece baskin harmonik siparisine dayanir; yon "
            "(radyal/aksiyal) olcumu ve faz bilgisi olmadan ayni baskin "
            "harmonige sahip birden fazla ariza tipi kesin olarak ayirt "
            "edilemez."
        ),
    }


def load_relative_phase_reference_table(chunks: list[dict]) -> list[dict]:
    """table_p61_0'i ("Table 6.1 Typical Table for Relative Phase Data",
    Bolum 6.2.2 Relative Phase Measurement) yapilandirilmis satirlara cevirir.

    ONEMLI KAPSAM NOTU: bu tablo bir ARIZA TANI ESIGI DEGIL. Kaynak
    bolumu (6.2 Dual and Multi-Channel Analysis > 6.2.2 Relative Phase
    Measurement) genel bir OLCUM/RAPORLAMA YONTEMI anlatiyor — "iki kanal
    arasindaki bagil fazi nasil olcer ve tablo halinde raporlarsin"
    sorusuna ornek veriyor, herhangi bir spesifik arizaya (hizasizlik
    dahil) ait dogrulanmis bir esik degeri sunmuyor. table_p61_0'daki
    sayisal degerler (87˚, 123˚, 165˚) bu yuzden fault_localization
    icinde bir PASS/FAIL esigi olarak degil, sadece BAGLAMSAL bir
    referans/ornek olarak kullanilmali — bkz. check_coupling_phase.
    """
    chunk = find_chunk(chunks, "table_p61_0")
    rows = []
    for order, ch1_amp, ch2_amp, phase in chunk["rows"]:
        order = order.strip()
        if not order or order in ("…", "..."):
            continue  
        rows.append({
            "order": order.upper(),
            "ch1_amplitude": float(ch1_amp) if ch1_amp else None,
            "ch2_amplitude": float(ch2_amp) if ch2_amp else None,
            "relative_phase_deg": float(phase.replace("˚", "").strip()) if phase else None,
        })
    return rows


