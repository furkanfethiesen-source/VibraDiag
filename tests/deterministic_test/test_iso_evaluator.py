"""
Unit tests for deterministic_tools.ıso_evaluator
=================================================
Tests the LangGraph ISO severity evaluator node, state updates,
and cached table execution.
"""

from __future__ import annotations

import importlib
import pytest

from src.deterministic_tools.json_rule_engine import load_severity_table

# Safe dynamic import in case of Unicode normalization variations across systems
iso_module = importlib.import_module("deterministic_tools.ıso_evaluator")
iso_evaluator_node = getattr(iso_module, "iso_evaluator_node")


class TestIsoEvaluatorNode:
    def test_iso_evaluator_node_with_cached_table(self, mock_table_chunks):
        severity_table = load_severity_table(mock_table_chunks)
        state = {
            "machine_class": "Class II",
            "vibration_velocity_rms_mm_s": 3.5,
            "severity_table": severity_table,
        }

        output = iso_evaluator_node(state)
        # Class II at 3.5 mm/s -> Zone C (2.8 - 7.1)
        assert output["iso_severity_zone"] == "C"
        assert output["iso_severity_meaning"] == "Just Tolerable"
        assert output["iso_severity_range_mm_s"] == (2.8, 7.1)
        assert "Medium machines" in output["iso_severity_machine_description"]
        assert output["severity_table"] is severity_table

    def test_iso_evaluator_node_zone_d_critical(self, mock_table_chunks):
        severity_table = load_severity_table(mock_table_chunks)
        state = {
            "machine_class": "Class I",
            "vibration_velocity_rms_mm_s": 8.0,
            "severity_table": severity_table,
        }

        output = iso_evaluator_node(state)
        # Class I >= 4.5 -> Zone D
        assert output["iso_severity_zone"] == "D"
        assert output["iso_severity_meaning"] == "Unacceptable"

    def test_iso_evaluator_node_negative_rms_raises(self, mock_table_chunks):
        severity_table = load_severity_table(mock_table_chunks)
        state = {
            "machine_class": "Class I",
            "vibration_velocity_rms_mm_s": -2.0,
            "severity_table": severity_table,
        }
        with pytest.raises(ValueError, match="rms_mm_s negatif olamaz"):
            iso_evaluator_node(state)
