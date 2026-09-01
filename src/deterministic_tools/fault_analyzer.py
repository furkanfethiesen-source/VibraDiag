"""
VibraDiag — Shared Fault Analysis & Diagnosis Helpers.
======================================================

Provides centralized fault extraction, scoring, and primary fault resolution
across Signal Processing Subgraph, DSP Consistency Checker, Prompt Builder,
and Live Test metadata extraction.
"""

from __future__ import annotations

from typing import Any


def pick_primary_fault(
    direct_spectrum_diagnosis: dict[str, Any] | None = None,
    envelope_diagnosis: dict[str, Any] | None = None,
    reciprocating_diagnosis: dict[str, Any] | None = None,
    channel_energy_map: dict[str, float] | None = None,
    min_confidence: float = 0.40,
    min_harmonics: int = 2,
) -> dict[str, Any]:
    """
    Extracts and ranks all candidate faults from direct spectrum, envelope (bearing),
    and reciprocating diagnoses, enforcing strict multi-harmonic / sideband criteria.

    Valid candidates with strong spectral evidence are placed into `all_faults`,
    while weak/isolated peaks with potential noise contamination are retained in `weak_candidates`.

    Returns
    -------
    dict[str, Any]
        {
            "primary_fault_name": str | None,
            "fault_type_abbr": str | None,
            "confidence": float,
            "severity": float,
            "score": float,
            "category": str | None,
            "sidebands_detected": bool,
            "all_faults": list[dict[str, Any]],
            "weak_candidates": list[dict[str, Any]],
        }
    """
    candidates: list[dict[str, Any]] = []
    weak_candidates: list[dict[str, Any]] = []

    if direct_spectrum_diagnosis and isinstance(direct_spectrum_diagnosis, dict):
        for fault_key, data in direct_spectrum_diagnosis.items():
            if fault_key == "visual_pattern_cross_check" or not isinstance(data, dict):
                continue

            conf = float(data.get("confidence", 0.0) or 0.0)
            sev = float(data.get("severity", 0.0) or 0.0)
            detected = bool(data.get("detected", conf > 0.0))

            if detected and (conf > 0.0 or sev > 0.0):
                fault_name = data.get("fault_name") or fault_key.replace("_", " ").title()
                abbr = {
                    "unbalance": "UNBALANCE",
                    "misalignment": "MISALIGNMENT",
                    "looseness": "LOOSENESS",
                }.get(fault_key.lower(), fault_key.upper())

                entry = {
                    "name": fault_name,
                    "abbr": abbr,
                    "confidence": conf,
                    "severity": sev,
                    "score": round(conf * (min(sev, 10.0) if sev > 0.0 else 1.0), 3),
                    "category": "direct_spectrum",
                    "sidebands_detected": False,
                    "details": data,
                }

                if conf >= 0.50 and sev >= 0.50:
                    candidates.append(entry)
                else:
                    weak_candidates.append(entry)

    if envelope_diagnosis and isinstance(envelope_diagnosis, dict):
        for ch_name, ch_faults in envelope_diagnosis.items():
            if not isinstance(ch_faults, dict):
                continue
            for fault_code, data in ch_faults.items():
                if not isinstance(data, dict):
                    continue

                conf = float(data.get("confidence", 0.0) or 0.0)
                sev = float(data.get("severity", 0.0) or 0.0)
                n_harmonics = int(data.get("n_harmonics_matched", len(data.get("matched_harmonics", []))) or 0)
                sideband_dict = data.get("sidebands", {}) if isinstance(data.get("sidebands"), dict) else {}
                sidebands_detected = bool(sideband_dict.get("sidebands_detected", False))

                if conf > 0.0 or sev > 0.0 or n_harmonics > 0:
                    fault_name_tr = {
                        "BPFO": "Dış Bilezik Arızası (BPFO)",
                        "BPFI": "İç Bilezik Arızası (BPFI)",
                        "BSF": "Bilya Arızası (BSF)",
                        "FTF": "Kafes Arızası (FTF)",
                    }.get(fault_code.upper(), f"Rulman {fault_code.upper()} Arızası")

                    sev_term = max(sev, 0.5) if sev > 0.0 else 1.0
                    base_score = conf * (sev_term + (n_harmonics * 1.5))

                    # Adaptive Sideband Family Energy (SFER) bonus
                    # For advanced/large flaws (Stage 3/4), modulation sideband energy carries critical diagnostic weight
                    harm_energy = sum(float(h.get("amplitude", 0.0) or 0.0) for h in data.get("matched_harmonics", []))
                    sb_energy = float(sideband_dict.get("total_sideband_energy", 0.0) or 0.0)
                    if sidebands_detected:
                        if harm_energy > 1e-6 and sb_energy > 0:
                            sfer = sb_energy / harm_energy
                            sb_multiplier = 1.0 + min(0.60, 0.35 + 0.25 * min(1.0, sfer))
                        else:
                            sb_multiplier = 1.35
                        score_val = base_score * sb_multiplier
                    else:
                        score_val = base_score

                    # Scale with absolute defect energy (harmonics + sidebands) to prevent quiet noise floor distortion
                    tot_defect_e = harm_energy + sb_energy
                    if tot_defect_e > 0:
                        score_val *= (1.0 + min(0.40, tot_defect_e * 8.0))

                    # FTF (Cage Fault) is almost always a modulation component when raceway/ball damage exists
                    if fault_code.upper() == "FTF":
                        score_val *= 0.35

                    # Channel weighting: Bearing housing sensors (DE, FE, main) are primary direct measurements.
                    # Base / foundation casing sensors (BA) measure indirect structural transmission and have higher background structural noise.
                    channel_weight = 0.70 if str(ch_name).upper() in ("BA", "BASE") else 1.0
                    if channel_energy_map and str(ch_name).upper() not in ("BA", "BASE"):
                        housing_energies = [v for k, v in channel_energy_map.items() if str(k).upper() not in ("BA", "BASE") and v > 0]
                        max_housing_energy = max(housing_energies, default=1.0)
                        ch_energy = float(channel_energy_map.get(ch_name, 0.0) or 0.0)
                        if max_housing_energy > 0 and ch_energy > 0:
                            import numpy as np
                            channel_weight *= float(np.sqrt(ch_energy / max_housing_energy))

                    final_score = round(score_val * channel_weight, 3)

                    entry = {
                        "name": f"{fault_name_tr} [{ch_name}]",
                        "abbr": fault_code.upper(),
                        "confidence": conf,
                        "severity": sev,
                        "score": final_score,
                        "category": "envelope",
                        "channel": ch_name,
                        "n_harmonics_matched": n_harmonics,
                        "sidebands_detected": sidebands_detected,
                        "sidebands": sideband_dict,
                        "details": data,
                    }

                    is_strong = (n_harmonics >= min_harmonics and conf >= min_confidence) or (
                        n_harmonics >= 1 and sidebands_detected and conf >= 0.33
                    )

                    if is_strong:
                        candidates.append(entry)
                    else:
                        weak_candidates.append(entry)

    if reciprocating_diagnosis and isinstance(reciprocating_diagnosis, dict):
        for fault_key, data in reciprocating_diagnosis.items():
            if not isinstance(data, dict):
                continue
            conf = float(data.get("confidence", 0.0) or 0.0)
            sev = float(data.get("severity", 0.0) or 0.0)
            if conf > 0.0 or sev > 0.0:
                fault_name = data.get("fault_name") or fault_key.replace("_", " ").title()
                entry = {
                    "name": fault_name,
                    "abbr": fault_key.upper(),
                    "confidence": conf,
                    "severity": sev,
                    "score": round(conf * (max(sev, 0.5) if sev > 0.0 else 1.0), 3),
                    "category": "reciprocating",
                    "sidebands_detected": False,
                    "n_harmonics_matched": 0,
                    "details": data,
                }
                if conf >= 0.50:
                    candidates.append(entry)
                else:
                    weak_candidates.append(entry)

    candidates.sort(key=lambda x: (x.get("score", 0.0), x.get("severity", 0.0)), reverse=True)
    weak_candidates.sort(key=lambda x: (x.get("score", 0.0), x.get("severity", 0.0)), reverse=True)

    if candidates:
        top = candidates[0]
        co_occurring = []
        is_compound = False
        secondary_name = None
        secondary_abbr = None

        if len(candidates) > 1:
            second = candidates[1]
            # Compound defect detection: e.g. Ball fault causing Inner Race modulation or vice-versa
            # If second candidate is a distinct fault type with high confidence (>=0.70) and score >= 65% of top score
            if second["abbr"] != top["abbr"] and second["score"] >= 0.65 * top["score"] and second["confidence"] >= 0.70:
                is_compound = True
                secondary_name = second["name"]
                secondary_abbr = second["abbr"]
                co_occurring.append(second)

        return {
            "primary_fault_name": top["name"],
            "fault_type_abbr": top["abbr"],
            "confidence": top["confidence"],
            "severity": top["severity"],
            "score": top["score"],
            "category": top["category"],
            "sidebands_detected": top.get("sidebands_detected", False),
            "n_harmonics_matched": int(top.get("n_harmonics_matched", 0) or 0),
            "dominant_channel": top.get("channel"),
            "is_compound_fault": is_compound,
            "secondary_fault_name": secondary_name,
            "secondary_fault_abbr": secondary_abbr,
            "co_occurring_faults": co_occurring,
            "all_faults": candidates,
            "weak_candidates": weak_candidates,
        }
    elif weak_candidates:
        top = weak_candidates[0]
        return {
            "primary_fault_name": f"{top['name']} (Düşük Güven)",
            "fault_type_abbr": top["abbr"],
            "confidence": top["confidence"],
            "severity": top["severity"],
            "score": top["score"],
            "category": top["category"],
            "sidebands_detected": top.get("sidebands_detected", False),
            "n_harmonics_matched": int(top.get("n_harmonics_matched", 0) or 0),
            "dominant_channel": top.get("channel"),
            "is_compound_fault": False,
            "secondary_fault_name": None,
            "secondary_fault_abbr": None,
            "co_occurring_faults": [],
            "all_faults": [],
            "weak_candidates": weak_candidates,
        }

    return {
        "primary_fault_name": None,
        "fault_type_abbr": None,
        "confidence": 0.0,
        "severity": 0.0,
        "score": 0.0,
        "category": None,
        "sidebands_detected": False,
        "n_harmonics_matched": 0,
        "dominant_channel": None,
        "is_compound_fault": False,
        "secondary_fault_name": None,
        "secondary_fault_abbr": None,
        "co_occurring_faults": [],
        "all_faults": [],
        "weak_candidates": [],
    }
