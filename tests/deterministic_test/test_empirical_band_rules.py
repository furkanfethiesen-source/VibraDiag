"""
Unit tests for deterministic_tools.empirical_band_rules
========================================================
Tests empirical envelope filter band selection based on RPM,
overlapping ranges, fallback behavior, and input validation.
"""

from __future__ import annotations

import pytest

from src.deterministic_tools.empirical_band_rules import select_band_empirical


class TestEmpiricalBandRules:
    def test_select_band_low_speed(self, mock_table_chunks):
        # 20 RPM -> Filter 1 (0-25 RPM, 5-100 Hz)
        res = select_band_empirical(rpm=20.0, chunks=mock_table_chunks)
        assert res["filter"] == 1
        assert res["band_hz"] == (5.0, 100.0)
        assert res["source"] == "table_p29_vision"

    def test_select_band_overlapping_prefers_lower(self, mock_table_chunks):
        # 25 RPM is in both Filter 1 (0-25) and Filter 2 (25-250)
        # Empirical rule specifies preferring the lower-frequency / lower-filter index
        res = select_band_empirical(rpm=25.0, chunks=mock_table_chunks)
        assert res["filter"] == 1

    def test_select_band_medium_speed(self, mock_table_chunks):
        # 1797 RPM -> Filter 3 (250-2500 RPM, 500-10000 Hz)
        res = select_band_empirical(rpm=1797.0, chunks=mock_table_chunks)
        assert res["filter"] == 3
        assert res["band_hz"] == (500.0, 10000.0)
        assert res["analyzing_range_hz"] == (100.0, 1000.0)

    def test_select_band_high_speed(self, mock_table_chunks):
        # 3600 RPM -> Filter 4 (2500-... RPM, 2500-... Hz)
        res = select_band_empirical(rpm=3600.0, chunks=mock_table_chunks)
        assert res["filter"] == 4
        assert res["band_hz"] == (2500.0, float("inf"))

    def test_select_band_invalid_rpm(self, mock_table_chunks):
        with pytest.raises(ValueError, match="rpm pozitif olmali"):
            select_band_empirical(rpm=0.0, chunks=mock_table_chunks)

        with pytest.raises(ValueError, match="rpm pozitif olmali"):
            select_band_empirical(rpm=-500.0, chunks=mock_table_chunks)

    def test_select_band_fallback_distance(self):
        # Gap between speed ranges where rpm=15 has no candidates
        gap_chunks = [
            {
                "chunk_id": "table_p29_vision",
                "content_type": "table",
                "rows": [
                    ["1", "5 – 100 Hz", "0 – 10 RPM", "10 – 50 Hz"],
                    ["2", "50 – 1000 Hz", "20 – 50 RPM", "50 – 200 Hz"],
                ],
            }
        ]
        # 15 RPM falls in the gap -> _distance chooses the closest
        res = select_band_empirical(rpm=15.0, chunks=gap_chunks)
        assert res["filter"] in (1, 2)
