
"""
VibraDiag - Coklu Kanal Demux & Sensor Fusion
================================================

Faz 2: readers.py'nin urettigi cok kanalli LoadedSignal (DE/FE/BA) uzerinde
calisan ariza lokalizasyon katmani.

Icerik
------
1) run_channel_pipeline / analyze_multichannel  
Her kanalda mevcut signal_processing.py zarf-analizi hattini
(select_band -> bandpass_filter -> hilbert_envelope -> envelope_fft
-> match_peaks) bagimsiz calistirir.

2) localize_fault                                
Kanal siddet (severity) skorlarini kiyaslayarak arizanin hangi
rulmana (DE/FE) fiziksel olarak daha yakin oldugunu tahmin eder.
Mantik: bir ic/dis bilezik arizasi, arizali rulmana en yakin
sensorde en yuksek genlikte gorulur (sinyal yolu/mesafe zayiflamasi).

3) check_coupling_phase                          
Iki kanal arasinda 1X/2X/3X bilesenlerinin bagil fazini hesaplar ve
~180 derece kaymayi (klasik kaplin hizasizligi imzasi) isaretler.

table_p61_0 ("Table 6.1 Typical Table for Relative Phase Data",
Bolum 6.2.2 Relative Phase Measurement) artik json_rule_engine
uzerinden okunuyor VE fonksiyona baglamsal referans olarak ekleniyor.
ONEMLI KAPSAM NOTU: bu tablo bir ARIZA TANI ESIGI DEGIL — kaynak
bolumu genel bir olcum/raporlama yontemi anlatiyor (misalignment'a
dogrulanmis bir esik sunmuyor). Bu yuzden tablodaki degerler
(87˚/123˚/165˚) PASS/FAIL kriteri olarak KULLANILMIYOR; sonuc sozlugune sadece "iste kaynaktaki tipik ornek boyleydi, kendi
olcumunle karsilastir" seklinde baglamsal bilgi olarak ekleniyor.
Asil pass/fail kriteri hala target_diff_deg=180/tolerance_deg=30
(genel literatur kabulu, tablo bunu degistirmedi cunku tablo esik
sunmuyor).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

from deterministic_tools.json_rule_engine import load_chunks, load_relative_phase_reference_table
from deterministic_tools.reader import LoadedSignal
from deterministic_tools.signal_processing import (
    bandpass_filter,
    calc_fault_freqs,
    envelope_fft,
    hilbert_envelope,
    is_kurtogram_reliable,
    match_peaks,
    select_band,
)


_CHANNEL_LOCATION_LABELS = {
    "DE": "Tahrik ucu (Drive End) rulmani",
    "FE": "Fan ucu (Fan End) rulmani",
    "BA": "Taban / govde (Base)",
    "main": "Tekil sensor",
}



@dataclass
class ChannelAnalysis:
    channel: str
    band: tuple
    fault_results: dict          
    envelope_freqs: np.ndarray
    envelope_magnitude: np.ndarray


def run_channel_pipeline(signal: np.ndarray, fs: float, fault_freqs: dict,
                        nlevels: int = 5, n_harmonics: int = 3,
                        tolerance_hz: float = 2.0,
                        override_band: tuple | None = None,
                        fr: float = 29.95) -> ChannelAnalysis:
    """Tek bir kanal icin tam zarf-analizi hattini calistirir.

    override_band verilirse select_band() (kurtogram) HIC calistirilmaz;
    dogrudan bu bant kullanilir. Faz 4'teki Fallback Node, kurtogram
    guvenilmez oldugunda table_p29_vision'dan sectigi bandi buraya
    override_band olarak geçirir."""
    if override_band is not None:
        band = override_band
    else:
        band, kurt = select_band(signal, fs, nlevels=nlevels)
        if not is_kurtogram_reliable(signal, fs, kurt):
            from deterministic_tools.empirical_band_rules import select_band_empirical
            band = select_band_empirical(fr * 60.0)["band_hz"]

    filtered = bandpass_filter(signal, fs, band)
    envelope = hilbert_envelope(filtered)
    freqs, magnitude = envelope_fft(envelope, fs)
    results = match_peaks(freqs, magnitude, fault_freqs,
                        n_harmonics=n_harmonics, tolerance_hz=tolerance_hz,
                        fr=fr)
    return ChannelAnalysis(
        channel="",
        band=band,
        fault_results=results,
        envelope_freqs=freqs,
        envelope_magnitude=magnitude,
    )


def analyze_multichannel(loaded: LoadedSignal, rpm: float, n_balls: int = 9,
                        ball_diameter: float = 7.94, pitch_diameter: float = 39.04,
                        contact_angle_deg: float = 0.0,
                        fe_bearing_params: dict | None = None,
                        n_harmonics: int = 3,
                        tolerance_hz: float = 2.0,
                        override_band: tuple | None = None) -> dict[str, ChannelAnalysis]:
    """LoadedSignal'daki her kanalda (DE/FE/BA/main) bagimsiz zarf analizi
    calistirir. DE/BA kanallari Drive End geometrisiyle (SKF 6205),
    FE kanali ise Fan End geometrisiyle (SKF 6203) taranir.

    override_band verilirse TUM kanallarda ayni bant kullanilir (her
    kanalda ayri kurtogram calismaz) — bkz. run_channel_pipeline."""
    fr = rpm / 60.0
    de_fault_freqs = calc_fault_freqs(
        rpm=rpm, n_balls=n_balls, ball_diameter=ball_diameter,
        pitch_diameter=pitch_diameter, contact_angle_deg=contact_angle_deg,
    )

    if fe_bearing_params and isinstance(fe_bearing_params, dict):
        fe_fault_freqs = calc_fault_freqs(
            rpm=rpm,
            n_balls=int(fe_bearing_params.get("n_balls", n_balls)),
            ball_diameter=float(fe_bearing_params.get("ball_diameter", ball_diameter)),
            pitch_diameter=float(fe_bearing_params.get("pitch_diameter", pitch_diameter)),
            contact_angle_deg=float(fe_bearing_params.get("contact_angle_deg", contact_angle_deg)),
        )
    else:
        fe_fault_freqs = de_fault_freqs

    results: dict[str, ChannelAnalysis] = {}
    for channel_name, signal in loaded.channels.items():
        ch_freqs = fe_fault_freqs if str(channel_name).upper() == "FE" else de_fault_freqs
        analysis = run_channel_pipeline(
            signal, loaded.fs, ch_freqs,
            n_harmonics=n_harmonics, tolerance_hz=tolerance_hz,
            override_band=override_band, fr=fr,
        )
        analysis.channel = channel_name
        results[channel_name] = analysis

    return results


def localize_fault(channel_analyses: dict[str, ChannelAnalysis],
                    severity_ratio_threshold: float = 1.5) -> dict:
    """Ayni ariza tipi (BPFO/BPFI/BSF/FTF) icin kanallar arasi siddet
    skorlarini kiyaslar. Bir kanaldaki siddet digerinden belirgin sekilde
    (severity_ratio_threshold kati) yuksekse, ariza o sensore fiziksel
    olarak daha yakin kabul edilir (sinyal yolu zayiflamasi mantigi).
    Fark yetersizse konum 'belirsiz' olarak isaretlenir — bu, tek kanaldan
    kesin lokalizasyon yapilamayacagi durumlar icin bilincli bir
    belirsizlik beyanidir, uydurma bir konum uretilmez."""
    fault_names: set[str] = set()
    for analysis in channel_analyses.values():
        fault_names.update(analysis.fault_results.keys())

    localization = {}
    for fault_name in fault_names:
        channel_severities = {
            ch: analysis.fault_results.get(fault_name, {}).get("severity", 0.0)
            for ch, analysis in channel_analyses.items()
        }
        ranked = sorted(channel_severities.items(), key=lambda kv: kv[1], reverse=True)
        top_channel, top_severity = ranked[0]
        second_severity = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_severity <= 0:
            dominant_channel = None
            location = "Tespit edilemedi (hicbir kanalda esik ustu siddet yok)"
            ratio = 0.0
        elif second_severity <= 0:
            dominant_channel = top_channel
            location = _CHANNEL_LOCATION_LABELS.get(top_channel, top_channel)
            ratio = float("inf")
        else:
            ratio = top_severity / second_severity
            if ratio >= severity_ratio_threshold:
                dominant_channel = top_channel
                location = _CHANNEL_LOCATION_LABELS.get(top_channel, top_channel)
            else:
                dominant_channel = None
                location = (
                    f"Belirsiz — kanallar arasi fark yetersiz "
                    f"(oran {ratio:.2f} < esik {severity_ratio_threshold})"
                )

        localization[fault_name] = {
            "dominant_channel": dominant_channel,
            "physical_location": location,
            "channel_severities": {k: round(v, 2) for k, v in channel_severities.items()},
            "severity_ratio": round(ratio, 2) if ratio != float("inf") else "inf",
        }

    return localization


def _phase_at_frequency(signal: np.ndarray, fs: float, target_freq: float) -> float:
    """Sinyalin belirli bir frekanstaki fazini (derece) FFT uzerinden okur."""
    n = len(signal)
    window = hann(n)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    idx = int(np.argmin(np.abs(freqs - target_freq)))
    return float(np.degrees(np.angle(spectrum[idx])))


_ORDER_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)X$", re.IGNORECASE)


def _order_to_harmonic(order: str) -> float:
    """'1X' -> 1.0, '2X' -> 2.0, '0.5X' -> 0.5"""
    match = _ORDER_PATTERN.match(order.strip())
    if not match:
        raise ValueError(f"Gecersiz order formati: {order!r} (orn. '1X', '2X' olmali)")
    return float(match.group(1))


def check_coupling_phase(channels: dict[str, np.ndarray], fs: float, rpm: float,
                        orders: list[str] | None = None,
                        tolerance_deg: float = 30.0,
                        target_diff_deg: float = 180.0,
                        reference_table: list[dict] | None = None,
                        table_chunks_path: str | None = None) -> dict:
    """Kanal ciftleri arasinda BIRDEN FAZLA order'da (varsayilan: 1X/2X/3X,
    table_p61_0'in raporlama kurgusuyla ayni) bagil faz kiyaslamasi yapar.
    Her order icin fark, target_diff_deg (varsayilan 180 derece) etrafinda
    tolerance_deg icindeyse kaplin hizasizligi imzasi olarak isaretlenir
    (matches_180_pattern).

    reference_table verilirse (verilmezse table_chunks_path'ten / varsayilan
    yoldan table_p61_0 otomatik yuklenir), her order'in SONUCUNA o order
    icin kaynak tablodaki tipik deger BAGLAMSAL BILGI olarak eklenir
    (reference_phase_deg_table_p61_0 alani).

    *** KAPSAM SINIRLAMASI: table_p61_0 ("Table 6.1 Typical Table for
    Relative Phase Data") bir ARIZA TANI ESIGI DEGIL — kaynak bolumu
    (6.2.2 Relative Phase Measurement) genel bir olcum/raporlama yontemi
    anlatiyor, misalignment'a ozgu dogrulanmis bir esik sunmuyor. Bu
    yuzden tablodaki degerler PASS/FAIL kararini ETKILEMEZ; sadece "kaynak
    kitaptaki tipik ornekte bu order'da soyle bir faz gorulmustu" seklinde
    referans bilgisi olarak sonuca eklenir. Asil pass/fail kriteri hala
    target_diff_deg/tolerance_deg (genel literatur kabulu). ***
    """
    if len(channels) < 2:
        raise ValueError("Faz kiyaslamasi icin en az 2 kanal gerekli.")

    if orders is None:
        orders = ["1X", "2X", "3X"]

    if reference_table is None:
        try:
            chunks = load_chunks(table_chunks_path) if table_chunks_path else load_chunks()
            reference_table = load_relative_phase_reference_table(chunks)
        except FileNotFoundError:
            reference_table = []  

    reference_by_order = {row["order"]: row["relative_phase_deg"] for row in reference_table}

    fr = rpm / 60.0
    results: dict[str, dict] = {}

    for order in orders:
        harmonic = _order_to_harmonic(order)
        target_freq = fr * harmonic
        phases = {name: _phase_at_frequency(sig, fs, target_freq) for name, sig in channels.items()}

        order_key = order.upper()
        pairs = {}
        for a, b in itertools.combinations(phases.keys(), 2):
            raw_diff = abs(phases[a] - phases[b]) % 360
            wrapped_diff = raw_diff if raw_diff <= 180 else 360 - raw_diff
            matches = abs(wrapped_diff - target_diff_deg) <= tolerance_deg

            pair_result = {
                f"phase_{a}_deg": round(phases[a], 1),
                f"phase_{b}_deg": round(phases[b], 1),
                "phase_diff_deg": round(wrapped_diff, 1),
                "matches_180_pattern": matches,
            }
            if order_key in reference_by_order:
                pair_result["reference_phase_deg_table_p61_0"] = reference_by_order[order_key]
                pair_result["reference_note"] = (
                    "Bu kaynak kitaptaki TIPIK ORNEK degeridir, dogrulanmis bir "
                    "esik degil — sadece baglamsal kiyaslama icindir."
                )

            pairs[f"{a}-{b}"] = pair_result

        results[order_key] = pairs

    return results


