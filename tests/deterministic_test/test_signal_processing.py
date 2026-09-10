"""
Unit tests for deterministic_tools.signal_processing
=====================================================
Tests bearing frequency calculation, bandpass filtering, Hilbert envelope,
FFT resolution, peak/sideband matching, kurtogram, and ratio patterns.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from src.deterministic_tools.signal_processing import (
    _check_sidebands,
    _dynamic_tolerance,
    _get_local_baseline,
    bandpass_filter,
    calc_fault_freqs,
    compute_cepstrum_spacing,
    envelope_fft,
    hilbert_envelope,
    is_kurtogram_reliable,
    match_named_frequencies,
    match_peaks,
    match_ratio_pattern,
    select_band,
)


class TestBearingFaultFreqs:
    def test_calc_fault_freqs_known_geometry(self):
        # SKF 6205 Bearing at 1797 RPM (~29.95 Hz)
        # N=9, d=7.94 mm, D=39.04 mm, theta=0
        freqs = calc_fault_freqs(
            rpm=1797.0,
            n_balls=9,
            ball_diameter=7.94,
            pitch_diameter=39.04,
            contact_angle_deg=0.0,
        )

        fr = 1797.0 / 60.0
        assert "BPFO" in freqs
        assert "BPFI" in freqs
        assert "BSF" in freqs
        assert "FTF" in freqs
        assert "2BSF" in freqs

        assert np.isclose(freqs["BPFO"], 107.36, atol=1.0)
        assert np.isclose(freqs["BPFI"], 162.19, atol=1.0)
        assert np.isclose(freqs["BSF"], 70.58, atol=1.0)
        assert np.isclose(freqs["FTF"], 11.93, atol=1.0)

        # Sum property: BPFO + BPFI should equal N * fr
        assert np.isclose(freqs["BPFO"] + freqs["BPFI"], 9 * fr, rtol=1e-5)

    def test_calc_fault_freqs_invalid_inputs(self):
        with pytest.raises(ValueError, match="n_balls en az 2 olmali"):
            calc_fault_freqs(rpm=1800.0, n_balls=1, ball_diameter=7.94, pitch_diameter=39.04)

        with pytest.raises(ValueError, match="ball_diameter ve pitch_diameter pozitif olmali"):
            calc_fault_freqs(rpm=1800.0, n_balls=9, ball_diameter=-1.0, pitch_diameter=39.04)

        with pytest.raises(ValueError, match="ball_diameter, pitch_diameter'dan kucuk olmali"):
            calc_fault_freqs(rpm=1800.0, n_balls=9, ball_diameter=50.0, pitch_diameter=39.04)


class TestFiltersAndEnvelope:
    def test_bandpass_filter_and_hilbert_envelope(self, synthetic_am_signal):
        fs = 12000.0
        sig = synthetic_am_signal(carrier_freq=2500.0, mod_freq=60.0, fs=fs, duration=1.0)

        # Apply bandpass around carrier
        filtered = bandpass_filter(sig, fs=fs, band=(2000.0, 3000.0))
        assert filtered.shape == sig.shape
        assert np.std(filtered) > 0.0

        # Hilbert envelope extracts the instantaneous amplitude (strictly non-negative)
        env = hilbert_envelope(filtered)
        assert env.shape == filtered.shape
        assert np.all(env >= 0.0)
        assert np.mean(env) > 0.0

        # FFT of envelope (remove_dc=True) should show a distinct peak at ~60 Hz
        freqs, mag = envelope_fft(env, fs=fs, zero_pad_factor=2, remove_dc=True)
        idx_peak = np.argmax(mag[freqs > 5.0])
        peak_freq = freqs[freqs > 5.0][idx_peak]
        assert np.isclose(peak_freq, 60.0, atol=2.0)

    def test_bandpass_filter_invalid_bands(self):
        sig = np.random.randn(1000)
        # band low >= high where high <= 1.0 cannot be recovered
        with pytest.raises(ValueError, match="Gecersiz bant"):
            bandpass_filter(sig, fs=10000.0, band=(1.0, 1.0))


class TestPeakAndSidebandMatching:
    def test_dynamic_tolerance(self):
        # harmonic=1, base_freq=100.0
        tol = _dynamic_tolerance(1, base_freq=100.0, min_abs_tol_hz=0.5, max_abs_tol_hz=2.5, tolerance_ratio=0.015)
        # target = 100 * 0.015 = 1.5, min(2.5, max(0.5, 1.5)) = 1.5
        assert np.isclose(tol, 1.5)

        # High harmonic cap at max_abs_tol_hz
        tol_high = _dynamic_tolerance(10, base_freq=100.0, max_abs_tol_hz=2.5)
        assert np.isclose(tol_high, 2.5)

    def test_get_local_baseline(self):
        freqs = np.linspace(0, 500, 2000)
        mag = np.full_like(freqs, 0.1)
        # Spike at 200 Hz
        mag[np.argmin(np.abs(freqs - 200.0))] = 5.0

        baseline = _get_local_baseline(freqs, mag, target_freq=200.0, fr=30.0)
        assert np.isclose(baseline, 0.1, atol=0.02)

    def test_check_sidebands_detected(self):
        freqs = np.linspace(0, 500, 2500)
        mag = np.full_like(freqs, 0.02)
        fr = 30.0
        center = 160.0

        # Center peak and sidebands at center +- 1X
        for f in [center, center - fr, center + fr]:
            idx = np.argmin(np.abs(freqs - f))
            mag[idx] = 1.0

        peak_idx, _ = find_peaks(mag, prominence=0.1)
        peak_freqs = freqs[peak_idx]

        res = _check_sidebands(
            freqs=freqs,
            magnitude=mag,
            peak_freqs=peak_freqs,
            center_freq=center,
            fr=fr,
            fault_name="BPFI",
        )
        assert res["sidebands_detected"] is True
        assert res["n_sidebands_matched"] >= 2
        assert res["total_sideband_energy"] > 0.0

    def test_match_peaks_bearing_detection(self):
        # Create synthetic envelope FFT with 3 harmonics of BPFI (160 Hz, 320 Hz, 480 Hz)
        freqs = np.linspace(0, 1000, 4000)
        mag = np.full_like(freqs, 0.05)  # noise floor

        bpfi = 160.0
        for h in [1, 2, 3]:
            idx = np.argmin(np.abs(freqs - (h * bpfi)))
            mag[idx] = 1.5 / h

        fault_freqs = {"BPFO": 105.0, "BPFI": 160.0, "BSF": 70.0, "FTF": 12.0}
        results = match_peaks(
            freqs=freqs,
            magnitude=mag,
            fault_freqs=fault_freqs,
            n_harmonics=3,
            fr=30.0,
        )

        assert "BPFI" in results
        bpfi_res = results["BPFI"]
        assert bpfi_res["n_harmonics_matched"] == 3
        assert bpfi_res["confidence"] > 0.70
        assert bpfi_res["severity"] > 1.0

        # BPFO should not have matched
        bpfo_res = results["BPFO"]
        assert bpfo_res["n_harmonics_matched"] == 0
        assert bpfo_res["confidence"] < 0.1


class TestKurtogramAndReliability:
    def test_is_kurtogram_reliable_with_clear_peak(self):
        rng = np.random.RandomState(42)
        sig = rng.normal(0, 1, 2000)
        # Add high impulsive shocks
        sig[500] = 50.0
        sig[1000] = 60.0
        _, kurtogram = select_band(sig, fs=8000.0, nlevels=3)
        reliable = is_kurtogram_reliable(sig, fs=8000.0, kurtogram=kurtogram, n_surrogates=3, rng=rng)
        assert isinstance(reliable, (bool, np.bool_))

    def test_select_band_returns_valid_tuple(self):
        t = np.linspace(0, 0.5, 4000)
        sig = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
        band, kurt_matrix = select_band(sig, fs=8000.0, nlevels=3)
        assert len(band) == 2
        assert band[0] < band[1]
        assert band[1] <= 4000.0
        assert len(kurt_matrix) > 0


class TestMechanicalRatioPatterns:
    def test_match_ratio_pattern_unbalance_and_misalignment(self):
        freqs = np.linspace(0, 300, 1500)
        mag = np.full_like(freqs, 0.02)
        fr = 30.0  # 1800 RPM

        # Strong 1X -> Unbalance
        idx_1x = np.argmin(np.abs(freqs - fr))
        mag[idx_1x] = 2.5

        res = match_ratio_pattern(freqs, mag, fr=fr)
        assert "unbalance" in res
        assert res["unbalance"]["confidence"] >= 0.50
        assert res["unbalance"]["severity"] > 1.0

        # Now add strong 2X -> Misalignment
        idx_2x = np.argmin(np.abs(freqs - (2 * fr)))
        mag[idx_2x] = 2.0
        res_mis = match_ratio_pattern(freqs, mag, fr=fr)
        assert "misalignment" in res_mis
        assert res_mis["misalignment"]["confidence"] >= 0.50

    def test_match_named_frequencies(self):
        freqs = np.linspace(0, 200, 1000)
        mag = np.full_like(freqs, 0.05)
        idx_50 = np.argmin(np.abs(freqs - 50.0))
        mag[idx_50] = 3.0

        matched = match_named_frequencies(freqs, mag, named_freqs={"line_freq": 50.0}, tolerance_hz=2.0)
        assert "line_freq" in matched
        assert matched["line_freq"]["matched"] is True
        assert matched["line_freq"]["amplitude"] > 0.2
        assert matched["line_freq"]["severity"] > 2.0


class TestCepstrumSpacing:
    def test_compute_cepstrum_spacing_synthetic(self):
        freqs = np.linspace(0, 1000, 2048)
        # Create multi-harmonic spectrum with 50 Hz spacing
        mag = np.full_like(freqs, 0.01)
        for harmonic in range(50, 800, 50):
            idx = np.argmin(np.abs(freqs - harmonic))
            mag[idx] = 1.0

        res = compute_cepstrum_spacing(freqs, mag, min_spacing_hz=15.0, max_spacing_hz=60.0)
        assert "dominant_spacing_hz" in res
        assert "cepstrum_peak_amp" in res
        assert res["dominant_spacing_hz"] is not None
        assert np.isclose(res["dominant_spacing_hz"], 50.0, atol=2.0)
