"""
Unit tests for deterministic_tools.fault_localization
======================================================
Tests single-channel pipeline, multi-channel analysis, physical location
inference based on sensor energy ratios, and coupling phase comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.deterministic_tools.fault_localization import (
    ChannelAnalysis,
    _order_to_harmonic,
    _phase_at_frequency,
    analyze_multichannel,
    check_coupling_phase,
    localize_fault,
    run_channel_pipeline,
)
from src.deterministic_tools.signal_processing import calc_fault_freqs


class TestSingleAndMultiChannelPipeline:
    def test_run_channel_pipeline_with_override_band(self, synthetic_sine):
        fs = 12000.0
        sig = synthetic_sine(freq=105.0, fs=fs, duration=0.5, amplitude=2.0)
        fault_freqs = calc_fault_freqs(rpm=1797.0, n_balls=9, ball_diameter=7.94, pitch_diameter=39.04)

        # Force override_band to bypass kurtogram calculation
        res = run_channel_pipeline(
            signal=sig,
            fs=fs,
            fault_freqs=fault_freqs,
            override_band=(50.0, 500.0),
            fr=1797.0 / 60.0,
        )

        assert isinstance(res, ChannelAnalysis)
        assert res.band == (50.0, 500.0)
        assert "BPFO" in res.fault_results
        assert len(res.envelope_freqs) > 0

    def test_analyze_multichannel(self, mock_loaded_multichannel):
        analyses = analyze_multichannel(
            loaded=mock_loaded_multichannel,
            rpm=1797.0,
            n_balls=9,
            ball_diameter=7.94,
            pitch_diameter=39.04,
            override_band=(50.0, 1000.0),
        )

        assert "DE" in analyses
        assert "FE" in analyses
        assert "BA" in analyses
        assert analyses["DE"].channel == "DE"


class TestFaultLocalization:
    def test_localize_fault_dominant_channel(self):
        # Create mock ChannelAnalyses where DE has BPFI severity 5.0, FE has 1.0
        freqs = np.linspace(0, 500, 100)
        mag = np.zeros_like(freqs)

        ch_analyses = {
            "DE": ChannelAnalysis(
                channel="DE",
                band=(1000, 2000),
                fault_results={"BPFI": {"severity": 5.0, "confidence": 0.8}},
                envelope_freqs=freqs,
                envelope_magnitude=mag,
            ),
            "FE": ChannelAnalysis(
                channel="FE",
                band=(1000, 2000),
                fault_results={"BPFI": {"severity": 1.0, "confidence": 0.3}},
                envelope_freqs=freqs,
                envelope_magnitude=mag,
            ),
        }

        loc = localize_fault(ch_analyses, severity_ratio_threshold=1.5)
        assert "BPFI" in loc
        bpfi_loc = loc["BPFI"]
        assert bpfi_loc["dominant_channel"] == "DE"
        assert "Drive End" in bpfi_loc["physical_location"]
        assert bpfi_loc["severity_ratio"] == 5.0

    def test_localize_fault_ambiguous_channel(self):
        freqs = np.linspace(0, 500, 100)
        mag = np.zeros_like(freqs)

        ch_analyses = {
            "DE": ChannelAnalysis(
                channel="DE",
                band=(1000, 2000),
                fault_results={"BPFI": {"severity": 2.2, "confidence": 0.8}},
                envelope_freqs=freqs,
                envelope_magnitude=mag,
            ),
            "FE": ChannelAnalysis(
                channel="FE",
                band=(1000, 2000),
                fault_results={"BPFI": {"severity": 2.0, "confidence": 0.7}},
                envelope_freqs=freqs,
                envelope_magnitude=mag,
            ),
        }

        loc = localize_fault(ch_analyses, severity_ratio_threshold=1.5)
        bpfi_loc = loc["BPFI"]
        assert "belirsiz" in bpfi_loc["physical_location"].lower()


class TestCouplingPhase:
    def test_order_to_harmonic(self):
        assert _order_to_harmonic("1X") == 1.0
        assert _order_to_harmonic("2x") == 2.0
        assert _order_to_harmonic("0.5X") == 0.5
        with pytest.raises(ValueError):
            _order_to_harmonic("invalid")

    def test_phase_at_frequency_and_coupling(self):
        fs = 10000.0
        duration = 1.0
        t = np.arange(0, duration, 1.0 / fs)
        freq = 30.0  # 1800 RPM 1X

        # sig1 is sin, sig2 is inverted sin (180 degree out of phase)
        sig1 = np.sin(2 * np.pi * freq * t)
        sig2 = -np.sin(2 * np.pi * freq * t)

        p1 = _phase_at_frequency(sig1, fs, freq)
        p2 = _phase_at_frequency(sig2, fs, freq)
        diff = abs(p1 - p2)
        # Should be ~180 degrees
        assert np.isclose(diff, 180.0, atol=5.0)

        # Coupling phase test
        channels = {"motor": sig1, "generator": sig2}
        coupling_res = check_coupling_phase(channels, fs=fs, rpm=1800.0, orders=["1X"])
        assert "1X" in coupling_res
        pair_data = coupling_res["1X"]["motor-generator"]
        assert pair_data["matches_180_pattern"] is True
        assert np.isclose(pair_data["phase_diff_deg"], 180.0, atol=10.0)

    def test_check_coupling_phase_validation(self):
        with pytest.raises(ValueError, match="en az 2 kanal gerekli"):
            check_coupling_phase({"main": np.array([1, 2, 3])}, fs=10000.0, rpm=1800.0)
