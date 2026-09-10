"""
Unit tests for deterministic_tools.fault_analyzer
==================================================
Tests pick_primary_fault ranking, multi-harmonic and sideband filtering,
SFER bonus calculation, weak candidate separation, and edge cases.
"""

from __future__ import annotations

from src.deterministic_tools.fault_analyzer import pick_primary_fault


class TestFaultAnalyzer:
    def test_pick_primary_fault_empty_inputs(self):
        res = pick_primary_fault()
        assert res["primary_fault_name"] is None
        assert res["fault_type_abbr"] is None
        assert res["confidence"] == 0.0
        assert res["all_faults"] == []
        assert res["weak_candidates"] == []

    def test_pick_primary_fault_envelope_bpfi_selection(self):
        envelope_diag = {
            "DE": {
                "BPFI": {
                    "confidence": 0.85,
                    "severity": 4.5,
                    "n_harmonics_matched": 3,
                    "matched_harmonics": [
                        {"harmonic": 1, "amplitude": 1.2},
                        {"harmonic": 2, "amplitude": 0.8},
                        {"harmonic": 3, "amplitude": 0.4},
                    ],
                    "sidebands": {
                        "sidebands_detected": True,
                        "total_sideband_energy": 0.6,
                    },
                },
                "BPFO": {
                    "confidence": 0.20,
                    "severity": 0.3,
                    "n_harmonics_matched": 1,
                    "matched_harmonics": [{"harmonic": 1, "amplitude": 0.1}],
                    "sidebands": {"sidebands_detected": False},
                },
            }
        }

        res = pick_primary_fault(envelope_diagnosis=envelope_diag)
        assert res["fault_type_abbr"] == "BPFI"
        assert "İç Bilezik" in res["primary_fault_name"]
        assert res["confidence"] == 0.85
        assert res["sidebands_detected"] is True
        assert len(res["all_faults"]) >= 1

        # BPFO should be in weak candidates because of low confidence/severity
        weak_abbrs = [w["abbr"] for w in res["weak_candidates"]]
        assert "BPFO" in weak_abbrs

    def test_pick_primary_fault_direct_spectrum_unbalance(self):
        direct_diag = {
            "unbalance": {
                "confidence": 0.90,
                "severity": 6.0,
                "detected": True,
            },
            "misalignment": {
                "confidence": 0.20,
                "severity": 0.2,
                "detected": False,
            },
        }

        res = pick_primary_fault(direct_spectrum_diagnosis=direct_diag)
        assert res["fault_type_abbr"] == "UNBALANCE"
        assert res["confidence"] == 0.90
        assert res["category"] == "direct_spectrum"

    def test_pick_primary_fault_reciprocating_diagnosis(self):
        recip_diag = {
            "combustion_imbalance": {
                "confidence": 0.80,
                "severity": 3.5,
                "detected": True,
                "fault_name": "Yanma Dengesizliği (0.5X)",
            }
        }

        res = pick_primary_fault(reciprocating_diagnosis=recip_diag)
        assert res["fault_type_abbr"] == "COMBUSTION_IMBALANCE"
        assert res["confidence"] == 0.80
        assert res["category"] == "reciprocating"

    def test_pick_primary_fault_competition_with_channel_energy(self):
        # Envelope BPFO vs Direct Spectrum Unbalance
        envelope_diag = {
            "DE": {
                "BPFO": {
                    "confidence": 0.75,
                    "severity": 3.0,
                    "n_harmonics_matched": 2,
                    "matched_harmonics": [
                        {"harmonic": 1, "amplitude": 0.9},
                        {"harmonic": 2, "amplitude": 0.5},
                    ],
                    "sidebands": {"sidebands_detected": False},
                }
            }
        }
        direct_diag = {
            "unbalance": {
                "confidence": 0.60,
                "severity": 1.2,
                "detected": True,
            }
        }
        channel_energy = {"DE": 1.5, "FE": 0.5}

        res = pick_primary_fault(
            direct_spectrum_diagnosis=direct_diag,
            envelope_diagnosis=envelope_diag,
            channel_energy_map=channel_energy,
        )
        assert res["fault_type_abbr"] == "BPFO"
        assert res["category"] == "envelope"
