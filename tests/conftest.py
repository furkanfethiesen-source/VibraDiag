"""
Pytest configuration and shared fixtures for VibraDiag deterministic tests.
===========================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from src.deterministic_tools.reader import LoadedSignal


@pytest.fixture
def synthetic_sine():
    """Generates a pure sine wave with optional DC offset and noise."""
    def _generator(
        freq: float = 30.0,
        fs: float = 12000.0,
        duration: float = 1.0,
        amplitude: float = 1.0,
        dc_offset: float = 0.0,
        noise_std: float = 0.0,
    ) -> np.ndarray:
        t = np.arange(0, duration, 1.0 / fs)
        sig = amplitude * np.sin(2.0 * np.pi * freq * t) + dc_offset
        if noise_std > 0:
            rng = np.random.default_rng(42)
            sig += rng.normal(0, noise_std, size=t.shape)
        return sig

    return _generator


@pytest.fixture
def synthetic_am_signal():
    """Generates an amplitude-modulated signal to test envelope extraction."""
    def _generator(
        carrier_freq: float = 2500.0,
        mod_freq: float = 65.0,
        fs: float = 12000.0,
        duration: float = 1.0,
        modulation_index: float = 0.7,
    ) -> np.ndarray:
        t = np.arange(0, duration, 1.0 / fs)
        carrier = np.sin(2.0 * np.pi * carrier_freq * t)
        modulator = 1.0 + modulation_index * np.sin(2.0 * np.pi * mod_freq * t)
        return carrier * modulator

    return _generator


@pytest.fixture
def mock_loaded_multichannel():
    """Returns a synthetic 3-channel (DE, FE, BA) LoadedSignal."""
    fs = 12000.0
    duration = 0.5
    t = np.arange(0, duration, 1.0 / fs)

    # DE has high energy fault pulse
    de_sig = 2.5 * np.sin(2.0 * np.pi * 105.0 * t) + 0.1 * np.random.default_rng(42).normal(size=t.shape)
    # FE has moderate energy
    fe_sig = 0.5 * np.sin(2.0 * np.pi * 105.0 * t) + 0.1 * np.random.default_rng(43).normal(size=t.shape)
    # BA has baseline noise
    ba_sig = 0.1 * np.random.default_rng(44).normal(size=t.shape)

    return LoadedSignal(
        channels={"DE": de_sig, "FE": fe_sig, "BA": ba_sig},
        fs=fs,
        rpm=1797.0,
        source_path="mock_cwru_105.mat",
        meta={"mock": True},
    )


@pytest.fixture
def mock_table_chunks():
    """Returns in-memory table_chunks mimicking table_p29_vision, fig_p15_0, and table_p47_0+p48."""
    return [
        {
            "chunk_id": "table_p29_vision",
            "content_type": "table",
            "rows": [
                ["1", "5 – 100 Hz", "0 – 25 RPM", "10 – 50 Hz"],
                ["2", "50 – 1000 Hz", "25 – 250 RPM", "50 – 200 Hz"],
                ["3", "500 – 10000 Hz", "250 – 2500 RPM", "100 – 1000 Hz"],
                ["4", "2500 – ... Hz", "2500 – ... RPM", "500 – 5000 Hz"],
            ],
        },
        {
            "chunk_id": "fig_p15_0",
            "content_type": "table",
            "structured_data": {
                "standard": "ISO 2372 / VDI 2056",
                "measurement_unit": "mm/s RMS",
                "zone_meanings": {
                    "A": "Good",
                    "B": "Satisfactory",
                    "C": "Just Tolerable",
                    "D": "Unacceptable",
                },
                "groups": [
                    {
                        "name": "Class I",
                        "machine_description": "Small machines up to 15 kW",
                        "foundation_type": "rigid",
                        "zones": {
                            "A": {"min_mm_s": 0.0, "max_mm_s": 0.71},
                            "B": {"min_mm_s": 0.71, "max_mm_s": 1.8},
                            "C": {"min_mm_s": 1.8, "max_mm_s": 4.5},
                            "D": {"min_mm_s": 4.5, "max_mm_s": None},
                        },
                    },
                    {
                        "name": "Class II",
                        "machine_description": "Medium machines 15-75 kW",
                        "foundation_type": "rigid",
                        "zones": {
                            "A": {"min_mm_s": 0.0, "max_mm_s": 1.12},
                            "B": {"min_mm_s": 1.12, "max_mm_s": 2.8},
                            "C": {"min_mm_s": 2.8, "max_mm_s": 7.1},
                            "D": {"min_mm_s": 7.1, "max_mm_s": None},
                        },
                    },
                    {
                        "name": "Class III",
                        "machine_description": "Large machines with rigid foundation",
                        "foundation_type": "rigid",
                        "zones": {
                            "A": {"min_mm_s": 0.0, "max_mm_s": 1.8},
                            "B": {"min_mm_s": 1.8, "max_mm_s": 4.5},
                            "C": {"min_mm_s": 4.5, "max_mm_s": 11.2},
                            "D": {"min_mm_s": 11.2, "max_mm_s": None},
                        },
                    },
                    {
                        "name": "Class IV",
                        "machine_description": "Large machines with soft foundation",
                        "foundation_type": "flexible",
                        "zones": {
                            "A": {"min_mm_s": 0.0, "max_mm_s": 2.8},
                            "B": {"min_mm_s": 2.8, "max_mm_s": 7.1},
                            "C": {"min_mm_s": 7.1, "max_mm_s": 18.0},
                            "D": {"min_mm_s": 18.0, "max_mm_s": None},
                        },
                    },
                ],
            },
        },
        {
            "chunk_id": "table_p47_0+p48",
            "content_type": "table",
            "rows": [
                [
                    "Power pulses",
                    "0.5 x RPM for 4-stroke engines 1 x RPM for 2-stroke engines",
                    "Combustion pressure unevenness",
                    "Balance fuel injection",
                ],
                [
                    "Piston slap",
                    "1 x RPM and multiples",
                    "Excessive piston to cylinder clearance",
                    "Inspect and replace liner or piston",
                ],
            ],
        },
    ]
