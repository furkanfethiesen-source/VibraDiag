"""
Unit tests for deterministic_tools.reciprocating_faults
========================================================
Tests reciprocating engine frequency calculations (stroke types, cylinder counts),
inertia harmonic generation, and diagnostic matching with cause/remedy texts.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.deterministic_tools.reciprocating_faults import (
    calc_reciprocating_freqs,
    diagnose_reciprocating,
    unbalance_inertia_harmonics,
)


class TestReciprocatingFaults:
    def test_calc_reciprocating_freqs_validations(self, mock_table_chunks):
        with pytest.raises(ValueError, match="stroke_type"):
            calc_reciprocating_freqs(rpm=1200.0, n_cylinders=4, stroke_type="6-stroke", chunks=mock_table_chunks)

        with pytest.raises(ValueError, match="n_cylinders"):
            calc_reciprocating_freqs(rpm=1200.0, n_cylinders=0, stroke_type="4-stroke", chunks=mock_table_chunks)

        with pytest.raises(ValueError, match="rpm pozitif olmali"):
            calc_reciprocating_freqs(rpm=-100.0, n_cylinders=4, stroke_type="4-stroke", chunks=mock_table_chunks)

    def test_calc_reciprocating_freqs_stroke_types(self, mock_table_chunks):
        # 1200 RPM -> fr = 20 Hz
        # 4 cylinders
        rpm = 1200.0
        fr = rpm / 60.0

        # In mock_table_chunks:
        # Power pulses: 0.5 x RPM for 4-stroke (which is 0.5 * 20 = 10 Hz)
        # Piston slap: 1 x RPM = 20 Hz
        freqs_4s = calc_reciprocating_freqs(rpm=rpm, n_cylinders=4, stroke_type="4-stroke", chunks=mock_table_chunks)
        assert "power_pulses" in freqs_4s
        assert np.isclose(freqs_4s["power_pulses"], 4 * fr)
        assert "piston_slap" in freqs_4s
        assert np.isclose(freqs_4s["piston_slap"], 1.0 * fr)

        # 2-stroke: 1 x RPM -> 20 Hz
        freqs_2s = calc_reciprocating_freqs(rpm=rpm, n_cylinders=4, stroke_type="2-stroke", chunks=mock_table_chunks)
        assert np.isclose(freqs_2s["power_pulses"], 2 * fr)

    def test_unbalance_inertia_harmonics(self):
        freqs = {"unbalance_inertia_forces": 40.0}
        harmonics = unbalance_inertia_harmonics(freqs, n_harmonics=3)
        assert harmonics == [40.0, 80.0, 120.0]

        with pytest.raises(ValueError, match="unbalance_inertia_forces"):
            unbalance_inertia_harmonics({}, n_harmonics=3)

        with pytest.raises(ValueError, match="n_harmonics en az 1 olmali"):
            unbalance_inertia_harmonics(freqs, n_harmonics=0)

    def test_diagnose_reciprocating(self, mock_table_chunks):
        rpm = 1200.0
        fr = rpm / 60.0  # 20 Hz
        freqs = np.linspace(0, 200, 1000)
        mag = np.full_like(freqs, 0.05)

        # Inject peak at 1X RPM = 20.0 Hz (piston slap in mock table)
        idx_20 = np.argmin(np.abs(freqs - 20.0))
        mag[idx_20] = 2.5

        diag = diagnose_reciprocating(
            freqs=freqs,
            magnitude=mag,
            rpm=rpm,
            n_cylinders=4,
            stroke_type="4-stroke",
            chunks=mock_table_chunks,
        )

        assert "piston_slap" in diag
        assert diag["piston_slap"]["matched"] is True
        assert "cause" in diag["piston_slap"]
        assert "remedy" in diag["piston_slap"]
