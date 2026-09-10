"""
Unit tests for deterministic_tools.machine_expert_nodes
========================================================
Tests LangGraph router functions and node execution for rotating
and reciprocating machinery expert pathways.
"""

from __future__ import annotations

import pytest

from src.deterministic_tools.machine_expert_nodes import (
    fallback_band_node,
    kurtogram_node,
    kurtogram_reliability_router,
    machine_type_router,
    reciprocating_expert_node,
    rotating_expert_node,
)
from src.deterministic_tools.reader import LoadedSignal


class TestMachineExpertRouters:
    def test_machine_type_router(self):
        assert machine_type_router({"machine_type": "rotating"}) == "rotating"
        assert machine_type_router({"machine_type": "reciprocating"}) == "reciprocating"
        assert machine_type_router({"machine_type": "Pistonlu Kompresör"}) == "reciprocating"
        assert machine_type_router({"machine_type": "unknown_type"}) == "rotating"
        assert machine_type_router({}) == "rotating"

    def test_kurtogram_reliability_router(self):
        assert kurtogram_reliability_router({"kurtogram_reliable": True}) == "rotating_expert"
        assert kurtogram_reliability_router({"kurtogram_reliable": False}) == "fallback_band"


class TestMachineExpertNodes:
    def test_kurtogram_node(self, mock_loaded_multichannel):
        state = {
            "loaded_signal": mock_loaded_multichannel,
            "rpm": 1797.0,
        }
        res = kurtogram_node(state)
        assert res["primary_channel"] == "DE"
        assert "kurtogram_band" in res
        assert "kurtogram_reliable" in res
        assert isinstance(res["kurtogram_reliable"], (bool, pytest.importorskip("numpy").bool_))

    def test_fallback_band_node(self):
        state = {"rpm": 1797.0}
        res = fallback_band_node(state)
        assert "final_band" in res
        assert res["band_source"] == "table_p29_vision"
        assert res["band_filter_used"] == 3

    def test_rotating_expert_node(self, mock_loaded_multichannel):
        state = {
            "loaded_signal": mock_loaded_multichannel,
            "primary_channel": "DE",
            "rpm": 1797.0,
            "n_balls": 9,
            "ball_diameter": 7.94,
            "pitch_diameter": 39.04,
            "contact_angle_deg": 0.0,
            "kurtogram_reliable": False,
            "final_band": (1000.0, 3000.0),
        }
        res = rotating_expert_node(state)
        assert "envelope_diagnosis" in res
        assert "fault_localization" in res
        assert "direct_spectrum_diagnosis" in res
        assert "time_domain_stats" in res

        stats = res["time_domain_stats"]
        assert stats["overall_rms"] > 0
        assert stats["crest_factor"] > 0

    def test_reciprocating_expert_node(self, mock_loaded_multichannel):
        state = {
            "loaded_signal": mock_loaded_multichannel,
            "rpm": 1200.0,
            "n_cylinders": 4,
            "stroke_type": "4-stroke",
        }
        res = reciprocating_expert_node(state)
        assert res["primary_channel"] == "DE"
        assert "reciprocating_diagnosis" in res
        assert "time_domain_stats" in res
