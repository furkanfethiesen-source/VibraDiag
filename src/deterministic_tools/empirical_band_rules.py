
"""
VibraDiag - Ampirik Zarf Filtresi Bant Kurallari (Fallback)
===============================================================

Kaynak: table_chunks.json / chunk_id="table_p29_vision"
"Table 3.2 Typical Filter Setting for Envelope Analysis"
(Bolum 3.8 Envelope Analysis, sayfa 29)

select_band()'in kurtogram tabanli otomatik bant secimi guvenilmez
oldugunda (bkz. signal_processing.is_kurtogram_reliable) devreye giren
ampirik/tablo-tabanli fallback.
"""

from __future__ import annotations

from deterministic_tools.json_rule_engine import load_chunks, load_envelope_filter_table


def select_band_empirical(rpm: float, chunks: list[dict] | None = None,
                        table_chunks_path: str | None = None) -> dict:
    """table_p29_vision'a gore (json_rule_engine uzerinden okunur) RPM'den
    ampirik zarf-filtresi bandi secer.

    NOT: RPM araliklari birbirini KESISIYOR (orn. 25-50 RPM hem Filtre 1
    hem Filtre 2'ye giriyor) — bu, kaynak tablonun kendi tasarimi (net
    cizgiler yerine yumusak gecis bolgeleri). Kesisim durumunda daha
    dusuk-frekansli (daha spesifik/dar) filtre tercih edilir.

    Parameters
    ----------
    rpm : float
        Sart devri (RPM)
    chunks : onceden yuklenmis table_chunks.json icerigi (verilmezse
        table_chunks_path'ten / varsayilan yoldan yuklenir)
    table_chunks_path : chunks verilmediyse kullanilacak JSON yolu

    Returns
    -------
    dict
        {"filter": int, "band_hz": (lo, hi), "analyzing_range_hz": (lo, hi),
        "source": "table_p29_vision"}
    """
    if rpm <= 0:
        raise ValueError("rpm pozitif olmali")

    if chunks is None:
        chunks = load_chunks(table_chunks_path) if table_chunks_path else load_chunks()
    filters = load_envelope_filter_table(chunks)

    candidates = [f for f in filters if f["speed_range_rpm"][0] <= rpm <= f["speed_range_rpm"][1]]

    if candidates:
        chosen = min(candidates, key=lambda f: f["filter_no"])  # dusuk-frekansli filtreyi tercih et
    else:
        def _distance(f):
            lo, hi = f["speed_range_rpm"]
            if rpm < lo:
                return lo - rpm
            if hi != float("inf") and rpm > hi:
                return rpm - hi
            return 0.0
        chosen = min(filters, key=_distance)

    return {
        "filter": chosen["filter_no"],
        "band_hz": chosen["freq_band_hz"],
        "analyzing_range_hz": chosen["analyzing_range_hz"],
        "source": "table_p29_vision",
    }


