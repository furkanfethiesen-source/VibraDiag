"""
Unit tests for deterministic_tools.signal_processing_subgraph
==============================================================
Tests LangGraph subgraph compilation, state mapping & sanitization,
kurtogram rectangularization, plot generation, and end-to-end execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.deterministic_tools.signal_processing_subgraph import (
    _rectangularize_kurtogram,
    _sanitize_for_state,
    build_signal_processing_subgraph,
    iso_check_node,
    iso_zone_router,
    map_parent_to_subgraph,
    map_subgraph_to_parent,
    plot_diagnostics_node,
    signal_processing_subgraph_node,
)


class TestSubgraphCompilationAndStateMapping:
    def test_build_signal_processing_subgraph(self):
        compiled = build_signal_processing_subgraph()
        assert compiled is not None
        # Graph should have nodes
        assert "iso_check" in compiled.nodes
        assert "rotating_expert" in compiled.nodes

    def test_rectangularize_kurtogram(self):
        # Ragged rows
        kurt = [
            [1.0, 2.0],
            [1.0, 3.0, 2.0, 4.0],
        ]
        rect, labels = _rectangularize_kurtogram(kurt)
        assert len(rect) == 2
        assert len(rect[0]) == 4
        assert len(rect[1]) == 4
        assert labels == ["Level 1", "Level 2"]

    def test_sanitize_for_state_numpy_primitives(self):
        data = {
            "arr": np.array([1, 2, 3]),
            "float_val": np.float64(3.14),
            "int_val": np.int32(42),
            "bool_val": np.bool_(True),
        }
        sanitized = _sanitize_for_state(data)
        assert isinstance(sanitized["arr"], list)
        assert isinstance(sanitized["float_val"], float)
        assert isinstance(sanitized["int_val"], int)
        assert isinstance(sanitized["bool_val"], bool)

    def test_map_parent_to_subgraph_and_back(self, mock_loaded_multichannel):
        parent_state = {
            "user_query": "Test query",
            "session_id": "sess_1",
            "loaded_signal": mock_loaded_multichannel,
            "rpm": 1797.0,
            "machine_metadata": {
                "machine_type": "rotating",
                "machine_class": "Class II",
            },
        }

        sub_input = map_parent_to_subgraph(parent_state)
        assert sub_input["machine_type"] == "rotating"
        assert sub_input["machine_class"] == "Class II"
        assert sub_input["rpm"] == 1797.0

        # Subgraph output mapping
        mock_sub_output = {
            "primary_channel": "DE",
            "rpm": 1797.0,
            "loaded_signal": mock_loaded_multichannel,  # Should be popped
            "kurtogram": [[1.0]],                       # Should be popped
            "direct_spectrum_diagnosis": {"unbalance": {"confidence": 0.5}},
        }
        parent_res = map_subgraph_to_parent(mock_sub_output)
        assert "signal_processing_result" in parent_res
        res_data = parent_res["signal_processing_result"]
        assert "loaded_signal" not in res_data
        assert "kurtogram" not in res_data
        assert res_data["primary_channel"] == "DE"


class TestIsoNodeAndZoneRouter:
    def test_iso_check_node_with_signal_rms(self, mock_loaded_multichannel):
        state = {
            "machine_class": "Class II",
            "loaded_signal": mock_loaded_multichannel,
            "primary_channel": "DE",
        }
        res = iso_check_node(state)
        assert "iso_severity_zone" in res
        assert "vibration_velocity_rms_mm_s" in res

    def test_iso_zone_router(self):
        assert iso_zone_router({"signal_processing_result": {"iso_severity_zone": "D"}}) == "emergency"
        assert iso_zone_router({"signal_processing_result": {"iso_severity_zone": "A"}}) == "continue"
        assert iso_zone_router({}) == "continue"


class TestPlotDiagnosticsNode:
    def test_plot_diagnostics_node(self, mock_loaded_multichannel):
        state = {
            "loaded_signal": mock_loaded_multichannel,
            "primary_channel": "DE",
            "rpm": 1797.0,
            "kurtogram": [[1.0, 2.0], [3.0, 4.0]],
            "envelope_diagnosis": {
                "DE": {
                    "BPFI": {
                        "confidence": 0.8,
                        "matched_harmonics": [{"harmonic": 1, "amplitude": 1.2}],
                    }
                }
            },
            "direct_spectrum_diagnosis": {},
        }
        plots = plot_diagnostics_node(state)
        assert "diagnostic_plots" in plots
        # diagnostic_plots may contain time_waveform, envelope_spectrum, kurtogram, etc.
        assert isinstance(plots["diagnostic_plots"], dict)


class TestFullSignalProcessingSubGraphNode:
    def test_signal_processing_subgraph_node_rotating(self, mock_loaded_multichannel):
        parent_state = {
            "loaded_signal": mock_loaded_multichannel,
            "machine_metadata": {
                "machine_type": "rotating",
                "machine_class": "Class I",
                "rpm": 1797.0,
            },
        }

        output = signal_processing_subgraph_node(parent_state)
        assert "signal_processing_result" in output
        assert "primary_fault" in output
        assert "primary_fault_data" in output
        assert "detected_rpm" in output
        assert output["detected_rpm"] == 1797.0

    def test_signal_processing_subgraph_node_reciprocating(self, mock_loaded_multichannel):
        parent_state = {
            "loaded_signal": mock_loaded_multichannel,
            "machine_metadata": {
                "machine_type": "reciprocating",
                "machine_class": "Class II",
                "rpm": 1200.0,
                "n_cylinders": 4,
            },
        }

        output = signal_processing_subgraph_node(parent_state)
        assert "signal_processing_result" in output
        assert "primary_fault_data" in output
        # Embedded RPM in mock_loaded_multichannel is 1797.0, which takes precedence
        assert output["detected_rpm"] == 1797.0
