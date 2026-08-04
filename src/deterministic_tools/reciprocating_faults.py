
"""
VibraDiag - Pistonlu Makine Ariza Frekanslari
================================================

Kaynak: table_chunks.json / chunk_id="table_p47_0+p48"
"Table 4.1 Predominant Frequency Components of Reciprocating Machines"
(Bolum 4.11 Reciprocating Machines Problems, sayfa 47-48)

Sebep/cozum (cause/remedy) METINLERI ise ISTISNA: JSON kaynagi Ingilizce
oldugu icin, kullanicinin ana dili Turkce olan raporlarda okunabilirlik
adina asagidaki _CAUSE_REMEDY_TR sozlugu KORUNDU (bu sabit bir ceviri
katmani, sayisal mantik degil — cevirinin kendisi JSON'dan turetilemez).
Ham Ingilizce metin de sonucta 'raw_cause_en'/'raw_remedy_en' olarak
saglaniyor, seffaflik icin.
"""

from __future__ import annotations

from deterministic_tools.json_rule_engine import load_chunks, parse_reciprocating_table

_CAUSE_REMEDY_TR: dict[str, tuple[str, str]] = {
    "inertia_1x": (
        "Piston ve biyel kolunun birincil atalet kuvvetleri",
        "Uygun tasarim ve denge kutleleri",
    ),
    "inertia_2x": (
        "Piston ve biyel kolunun ikincil atalet kuvvetleri",
        "Uygun tasarim ve denge kutleleri",
    ),
    "power_pulses": (
        "Motor guc cevrimi",
        "Silindir sayisini artirmak",
    ),
    "misfiring_piston": (
        "Negatif guc stroku",
        "Motoru tamir et",
    ),
    "worn_connecting_rod_bearings": (
        "Piston yon degistirirken yatak darbesi",
        "Motoru tamir et",
    ),
    "worn_crankshaft_main_bearings": (
        "Her guc strokunda yatak darbesi",
        "Motoru tamir et",
    ),
    "piston_slap": (
        "Agir yuk altinda, biyel koluna dik kuvvet bileseni",
        "Motoru tamir et",
    ),
    "unbalance_inertia_forces": (
        "Yanlis duzeltme agirliklarindan kaynaklanan ikincil atalet kuvvetleri",
        "Uygun yedek parca secimi",
    ),
}

_DESCRIPTION_TO_KEY = {
    "Power pulses": "power_pulses",
    "Misfiring piston": "misfiring_piston",
    "Worn Connecting rod bearings": "worn_connecting_rod_bearings",
    "Worn crankshaft main bearings": "worn_crankshaft_main_bearings",
    "Piston slap": "piston_slap",
    "Unbalance inertia forces": "unbalance_inertia_forces",
}


def calc_reciprocating_freqs(rpm: float, n_cylinders: int,
                            stroke_type: str = "4-stroke",
                            chunks: list[dict] | None = None,
                            table_chunks_path: str | None = None) -> dict[str, float]:
    """table_p47_0+p48'i (json_rule_engine uzerinden) okuyup RPM ve
    silindir sayisindan yedi mekanizmanin beklenen frekanslarini (Hz)
    hesaplar.

    Parameters
    ----------
    rpm, n_cylinders, stroke_type : bkz. onceki versiyon
    chunks : onceden yuklenmis table_chunks.json icerigi (verilmezse
        table_chunks_path'ten / varsayilan yoldan yuklenir — tekrar
        tekrar dosya okumamak icin caller'in chunks'i cache'leyip
        gecirmesi onerilir, orn. LangGraph state'inde).
    table_chunks_path : chunks verilmediyse kullanilacak JSON yolu
        (varsayilan: json_rule_engine._DEFAULT_TABLE_CHUNKS_PATH)
    """
    if stroke_type not in ("4-stroke", "2-stroke"):
        raise ValueError("stroke_type '4-stroke' veya '2-stroke' olmali")
    if n_cylinders < 1:
        raise ValueError("n_cylinders en az 1 olmali")
    if rpm <= 0:
        raise ValueError("rpm pozitif olmali")

    if chunks is None:
        chunks = load_chunks(table_chunks_path) if table_chunks_path else load_chunks()
    rows = parse_reciprocating_table(chunks)

    fr = rpm / 60.0
    freqs: dict[str, float] = {}
    pending_references: dict[str, str] = {}

    for row in rows:
        desc = row["description"]
        mult = row["multipliers"]

        if desc == "Inertia forces":
            for v in mult.get("any", []):
                if v == 1.0:
                    freqs["inertia_1x"] = 1.0 * fr
                elif v == 2.0:
                    freqs["inertia_2x"] = 2.0 * fr
            continue

        key = _DESCRIPTION_TO_KEY.get(desc)
        if key is None:
            continue  

        if "reference" in mult:
            pending_references[key] = "power_pulses"
        elif "4-stroke" in mult or "2-stroke" in mult:
            token = mult.get(stroke_type)
            if token is None:
                continue
            if token == "N":
                value = float(n_cylinders)
            elif token == "N/2":
                value = n_cylinders / 2.0
            else:
                value = float(token)
            freqs[key] = value * fr
        elif mult.get("any"):
            freqs[key] = mult["any"][0] * fr

    for key, ref_key in pending_references.items():
        if ref_key in freqs:
            freqs[key] = freqs[ref_key]

    return freqs


def unbalance_inertia_harmonics(freqs: dict[str, float], n_harmonics: int = 3) -> list[float]:
    """calc_reciprocating_freqs() ciktisindaki 'unbalance_inertia_forces'
    (tabloda '2 x RPM AND MULTIPLES') temel alinarak ust katlarini
    (2x, 4x, 6x, ...) uretir."""
    if "unbalance_inertia_forces" not in freqs:
        raise ValueError("freqs icinde 'unbalance_inertia_forces' yok — once calc_reciprocating_freqs cagirilmali")
    if n_harmonics < 1:
        raise ValueError("n_harmonics en az 1 olmali")
    base = freqs["unbalance_inertia_forces"]
    return [base * k for k in range(1, n_harmonics + 1)]


def diagnose_reciprocating(freqs, magnitude, rpm: float, n_cylinders: int,
                            stroke_type: str = "4-stroke",
                            tolerance_hz: float = 2.0,
                            chunks: list[dict] | None = None) -> dict:
    """calc_reciprocating_freqs() ile hesaplanan (JSON-tabanli) frekanslari
    spektrumla eslestirir (signal_processing.match_named_frequencies
    uzerinden) ve her mekanizma icin Turkce sebep/cozum metnini ekler.

    NOT: worn_connecting_rod_bearings, piston_slap ve unbalance_inertia_forces
    tabloda AYNI frekansta (2x RPM) gorunur — spektrumda bu uc mekanizma
    ayni tepe noktasini paylasir ve tek basina spektral eslestirmeyle
    ayristirilamaz (tablonun kendi bilinen limitasyonu). Ayirt etmek icin
    zaman dalga formu (piston slap darbe seklinde gorunur) veya faz analizi
    gerekir.
    """
    from .signal_processing import match_named_frequencies

    if chunks is None:
        chunks = load_chunks()

    named_freqs = calc_reciprocating_freqs(rpm, n_cylinders, stroke_type, chunks=chunks)
    results = match_named_frequencies(freqs, magnitude, named_freqs, tolerance_hz=tolerance_hz)

    for name, (cause_tr, remedy_tr) in _CAUSE_REMEDY_TR.items():
        if name in results:
            results[name]["cause"] = cause_tr
            results[name]["remedy"] = remedy_tr
            results[name]["source"] = "table_p47_0+p48"

    return results

