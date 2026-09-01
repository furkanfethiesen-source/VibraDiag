"""
VibraDiag — Labeled Datasets DSP Accuracy Benchmark & CSV Reporter
===================================================================

Bu script `data/signal_test/` altındaki tüm `.mat` uzantılı test sinyallerini
deterministik DSP hattından (`analyze_multichannel` ve `pick_primary_fault`)
geçirir ve detaylı teşhis metriklerini `data/deterministic_tool_test/dsp_accuracy.csv`
dosyasına kaydeder.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

# Add src to sys.path if not present
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np

from deterministic_tools.fault_analyzer import pick_primary_fault
from deterministic_tools.fault_localization import analyze_multichannel
from deterministic_tools.reader import SignalReaderFactory


# CWRU Bilinen Referans Zemin Gerçekleri (Ground Truth)
KNOWN_GROUND_TRUTH: dict[str, dict[str, Any]] = {
    # BPFI - İç Bilezik Arızası
    "105.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.007\" Inner Race Fault @ 1797 RPM"},
    "108.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.007\" Inner Race Fault @ 1730 RPM"},
    "120.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.014\" Inner Race Fault @ 1748 RPM"},
    "170.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.021\" Inner Race Fault @ 1772 RPM"},
    "172.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.021\" Inner Race Fault @ 1730 RPM"},
    "211.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.028\" Inner Race Fault @ 1750 RPM"},
    "3001.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.028\" Inner Race Fault @ 1797 RPM"},
    "3004.mat": {"expected_fault": "BPFI", "expected_channel": "DE", "description": "CWRU 12k DE 0.028\" Inner Race Fault @ 1730 RPM"},

    # BSF - Bilye Arızası
    "119.mat": {"expected_fault": "BSF",  "expected_channel": "DE", "description": "CWRU 12k DE 0.007\" Ball Fault @ 1772 RPM"},
    "185.mat": {"expected_fault": "BSF",  "expected_channel": "DE", "description": "CWRU 12k DE 0.014\" Ball Fault @ 1797 RPM"},
    "187.mat": {"expected_fault": "BSF",  "expected_channel": "DE", "description": "CWRU 12k DE 0.014\" Ball Fault @ 1749 RPM"},
    "223.mat": {"expected_fault": "BSF",  "expected_channel": "DE", "description": "CWRU 12k DE 0.021\" Ball Fault @ 1772 RPM"},
    "3006.mat": {"expected_fault": "BSF",  "expected_channel": "DE", "description": "CWRU 12k DE 0.028\" Ball Fault @ 1772 RPM"},
    "3007.mat": {"expected_fault": "BSF",  "expected_channel": "DE", "description": "CWRU 12k DE 0.028\" Ball Fault @ 1750 RPM"},

    # BPFO - Dış Bilezik Arızası
    "131.mat": {"expected_fault": "BPFO", "expected_channel": "DE", "description": "CWRU 12k DE 0.007\" Outer Race Fault @ 1772 RPM"},
    "133.mat": {"expected_fault": "BPFO", "expected_channel": "DE", "description": "CWRU 12k DE 0.007\" Outer Race Fault @ 1725 RPM"},
    "146.mat": {"expected_fault": "BPFO", "expected_channel": "DE", "description": "CWRU 12k DE 0.014\" Outer Race Fault @ 1772 RPM"},
    "197.mat": {"expected_fault": "BPFO", "expected_channel": "DE", "description": "CWRU 12k DE 0.014\" Outer Race Fault @ 1797 RPM"},
    "235.mat": {"expected_fault": "BPFO", "expected_channel": "DE", "description": "CWRU 12k DE 0.021\" Outer Race Fault @ 1772 RPM"},
    "247.mat": {"expected_fault": "BPFO", "expected_channel": "DE", "description": "CWRU 12k DE 0.028\" Outer Race Fault @ 1772 RPM"},
}


def run_benchmark(
    signals_dir: str | Path = "data/signal_test",
    output_csv_path: str | Path = "data/deterministic_tool_test/dsp_accuracy.csv",
    n_balls: int = 9,
    ball_diameter: float = 7.94,
    pitch_diameter: float = 39.04,
    contact_angle_deg: float = 0.0,
) -> list[dict[str, Any]]:
    """Tüm test sinyallerini işler ve CSV çıktısı üretir."""
    signals_path = Path(signals_dir)
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(list(signals_path.glob("*.mat")))
    if not mat_files:
        print(f"UYARI: {signals_path} dizininde .mat dosyası bulunamadı.")
        return []

    records: list[dict[str, Any]] = []

    print(f"\n🚀 VibraDiag DSP Benchmark Başlatılıyor ({len(mat_files)} dosya)...")
    print("-" * 80)

    for fpath in mat_files:
        fname = fpath.name
        gt = KNOWN_GROUND_TRUTH.get(fname, {"expected_fault": "UNKNOWN", "expected_channel": "DE"})
        expected_abbr = gt.get("expected_fault", "UNKNOWN")

        # 1. Sinyal Yükleme
        try:
            loaded = SignalReaderFactory.load(str(fpath), expected_fs=12000.0)
        except Exception as e:
            print(f"❌ {fname} yüklenirken hata: {e}")
            continue

        fs = float(loaded.fs or 12000.0)
        rpm = float(loaded.rpm or 1749.0)

        # 2. Kanal Fiziksel Enerji Haritası (RMS & Peak)
        rms_map = {ch: float(np.sqrt(np.mean(sig**2))) for ch, sig in loaded.channels.items()}
        rms_str = ", ".join([f"{ch}:{v:.4f}" for ch, v in rms_map.items()])

        # 3. Çok Kanallı Zarf Analizi
        analyses = analyze_multichannel(
            loaded=loaded,
            rpm=rpm,
            n_balls=n_balls,
            ball_diameter=ball_diameter,
            pitch_diameter=pitch_diameter,
            contact_angle_deg=contact_angle_deg,
            override_band=None,
        )

        envelope_diag = {ch: a.fault_results for ch, a in analyses.items()}

        # 4. Birincil Arıza Çözümleme (Kanal Enerji Ağırlıklı & Adaptif SFER)
        primary = pick_primary_fault(
            direct_spectrum_diagnosis={},
            envelope_diagnosis=envelope_diag,
            reciprocating_diagnosis={},
            channel_energy_map=rms_map,
        )

        detected_name = primary.get("primary_fault_name") or "Tespit Edilemedi"
        detected_abbr = primary.get("fault_type_abbr") or "NONE"
        dominant_ch = primary.get("dominant_channel") or "DE"
        conf = float(primary.get("confidence", 0.0) or 0.0)
        sev = float(primary.get("severity", 0.0) or 0.0)
        final_score = float(primary.get("score", 0.0) or 0.0)

        is_compound = bool(primary.get("is_compound_fault", False))
        secondary_name = primary.get("secondary_fault_name") or "None"
        secondary_abbr = primary.get("secondary_fault_abbr") or "NONE"
        co_occurring = primary.get("co_occurring_faults", [])
        co_abbrs = [c.get("abbr") for c in co_occurring]

        # 5. Eşleşen Harmonikler ve Yan Bant Detayları
        dom_analysis = analyses.get(dominant_ch)
        filter_band_used = f"{dom_analysis.band[0]:.1f}-{dom_analysis.band[1]:.1f} Hz" if dom_analysis else "N/A"
        
        fault_detail = dom_analysis.fault_results.get(detected_abbr, {}) if dom_analysis else {}
        matched_harmonics_list = fault_detail.get("matched_harmonics", [])
        
        harm_str = ", ".join([f"{h.get('harmonic')}X:{h.get('found_hz', 0.0):.2f}Hz" for h in matched_harmonics_list]) or "None"
        amps_list = [round(float(h.get("amplitude", 0.0) or 0.0), 5) for h in matched_harmonics_list]
        
        sideband_info = fault_detail.get("sidebands", {})
        sidebands_detected = bool(sideband_info.get("sidebands_detected", False))
        sideband_count = int(sideband_info.get("n_sidebands_matched", 0) or 0)
        sideband_energy = float(sideband_info.get("total_sideband_energy", 0.0) or 0.0)

        # Cepstrum yan bant aralığı tahmini
        cep_spacing = None
        if dom_analysis and len(dom_analysis.envelope_freqs) > 0:
            from deterministic_tools.signal_processing import compute_cepstrum_spacing
            cep_res = compute_cepstrum_spacing(dom_analysis.envelope_freqs, dom_analysis.envelope_magnitude)
            cep_spacing = cep_res.get("dominant_spacing_hz")

        # Doğrulama Durumu
        if expected_abbr != "UNKNOWN":
            if detected_abbr == expected_abbr:
                status = "MATCH"
            elif secondary_abbr == expected_abbr or expected_abbr in co_abbrs:
                status = "COMPOUND_MATCH"
            else:
                status = "MISMATCH"
        else:
            status = "UNKNOWN"

        record = {
            "file_name": fname,
            "rpm": rpm,
            "sampling_rate_hz": fs,
            "expected_fault": expected_abbr,
            "detected_primary_fault": detected_name,
            "fault_type_abbr": detected_abbr,
            "dominant_channel": dominant_ch,
            "confidence": conf,
            "severity": sev,
            "final_score": final_score,
            "is_compound_fault": is_compound,
            "secondary_fault": secondary_name,
            "secondary_fault_abbr": secondary_abbr,
            "matched_harmonics": harm_str,
            "harmonic_amplitudes": json.dumps(amps_list),
            "sidebands_detected": sidebands_detected,
            "sideband_count": sideband_count,
            "sideband_energy": round(sideband_energy, 5),
            "cepstrum_spacing_hz": cep_spacing,
            "filter_band_used": filter_band_used,
            "channel_rms_levels": rms_str,
            "status": status,
        }
        records.append(record)

        icon = "✅" if status in ("MATCH", "COMPOUND_MATCH") else "⚠️"
        comp_str = f" + Eşlik Eden: {secondary_abbr}" if is_compound else ""
        print(f"{icon} {fname:10s} | Beklenen: {expected_abbr:4s} | Tespit: {detected_abbr:4s}{comp_str} [{dominant_ch}] | Güven: {conf:.2f} | Şiddet: {sev:.2f} | Skor: {final_score:.3f} | Durum: {status}")

    # CSV Kaydetme
    if records:
        fieldnames = list(records[0].keys())
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print("-" * 80)
        print(f"✅ Rapor başarıyla kaydedildi: {out_csv.resolve()}\n")

    return records


if __name__ == "__main__":
    run_benchmark()
