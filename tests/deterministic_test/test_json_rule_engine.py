"""
Unit tests for deterministic_tools.json_rule_engine
====================================================
Tests machine class normalization, ISO severity zone mapping,
text/range parsers, reciprocating table parsing, and error conditions.
"""

from __future__ import annotations

import pytest

from src.deterministic_tools.json_rule_engine import (
    classify_severity,
    clean_text,
    extract_all_multipliers,
    find_chunk,
    load_envelope_filter_table,
    load_severity_table,
    normalize_machine_class,
    parse_range,
    parse_reciprocating_table,
    parse_stroke_dependent_multipliers,
)


class TestMachineClassNormalization:
    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("Class I", "Class I"),
            ("class 1", "Class I"),
            ("1", "Class I"),
            (1, "Class I"),
            ("II", "Class II"),
            ("Class-2", "Class II"),
            ("class_3", "Class III"),
            (4, "Class IV"),
            ("iv", "Class IV"),
        ],
    )
    def test_normalize_valid_inputs(self, input_val, expected):
        assert normalize_machine_class(input_val) == expected

    @pytest.mark.parametrize("invalid_input", [0, 5, -1, "Class 5", "unknown", "V"])
    def test_normalize_invalid_inputs(self, invalid_input):
        with pytest.raises(ValueError):
            normalize_machine_class(invalid_input)


class TestSeverityClassification:
    def test_classify_severity_zones(self, mock_table_chunks):
        severity_table = load_severity_table(mock_table_chunks)

        # Class I: A: [0, 0.71), B: [0.71, 1.8), C: [1.8, 4.5), D: >= 4.5
        res_a = classify_severity("Class I", 0.5, severity_table)
        assert res_a["zone"] == "A"
        assert res_a["meaning"] == "Good"

        res_b = classify_severity("1", 1.0, severity_table)
        assert res_b["zone"] == "B"
        assert res_b["meaning"] == "Satisfactory"

        res_c = classify_severity("Class I", 2.5, severity_table)
        assert res_c["zone"] == "C"
        assert res_c["meaning"] == "Just Tolerable"

        res_d = classify_severity("Class I", 5.0, severity_table)
        assert res_d["zone"] == "D"
        assert res_d["meaning"] == "Unacceptable"

    def test_classify_negative_rms_error(self, mock_table_chunks):
        severity_table = load_severity_table(mock_table_chunks)
        with pytest.raises(ValueError, match="rms_mm_s negatif olamaz"):
            classify_severity("Class I", -1.0, severity_table)

    def test_classify_unknown_class_error(self, mock_table_chunks):
        severity_table = load_severity_table(mock_table_chunks)
        with pytest.raises(ValueError):
            classify_severity("Class 99", 2.0, severity_table)


class TestParsersAndTableLoaders:
    def test_find_chunk_success_and_fail(self, mock_table_chunks):
        chunk = find_chunk(mock_table_chunks, "table_p29_vision")
        assert chunk["chunk_id"] == "table_p29_vision"

        with pytest.raises(ValueError, match="bulunamadi"):
            find_chunk(mock_table_chunks, "non_existent_chunk_id")

    def test_parse_range(self):
        low, high = parse_range("5 – 100 Hz")
        assert (low, high) == (5.0, 100.0)

        low_inf, high_inf = parse_range("2,500 – ... RPM")
        assert (low_inf, high_inf) == (2500.0, float("inf"))

        with pytest.raises(ValueError):
            parse_range("single_number")

    def test_clean_text(self):
        text = "Multi-line\nwith × sign  and   spaces"
        cleaned = clean_text(text)
        assert cleaned == "Multi-line with x sign and spaces"

    def test_extract_all_multipliers(self):
        text = "are 1x and 2xRPM and 3.5 x RPM"
        multipliers = extract_all_multipliers(text)
        assert multipliers == [1.0, 2.0, 3.5]

    def test_parse_stroke_dependent_multipliers(self):
        text = "0.5 x RPM for 4-stroke engines 1 x RPM for 2-stroke engines"
        parsed = parse_stroke_dependent_multipliers(text)
        assert parsed == {"4-stroke": 0.5, "2-stroke": 1.0}

    def test_load_envelope_filter_table(self, mock_table_chunks):
        filters = load_envelope_filter_table(mock_table_chunks)
        assert len(filters) == 4
        assert filters[0]["filter_no"] == 1
        assert filters[0]["freq_band_hz"] == (5.0, 100.0)
        assert filters[3]["freq_band_hz"] == (2500.0, float("inf"))

    def test_parse_reciprocating_table(self, mock_table_chunks):
        rows = parse_reciprocating_table(mock_table_chunks)
        assert len(rows) == 2
        assert "Power pulses" in rows[0]["description"]
        assert rows[0]["multipliers"] == {"4-stroke": "N", "2-stroke": "N/2"}
        assert rows[1]["multipliers"] == {"any": [1.0]}
