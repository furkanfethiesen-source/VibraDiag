"""
VibraDiag - Signal Processing SubGraph ve İki Seviyeli State Katmanı
=====================================================================

Bu modül, deterministik sinyal işleme adımlarını (kurtogram, fallback band,
rotating/reciprocating expert, ISO değerlendirmesi) tek bir derlenmiş alt-grafta toplar.

Global state (VibraDiagMainState) ve local subgraph state (SignalProcessingInternalState)
tanımları `schemas.states` modülünden alınır.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from deterministic_tools.ıso_evaluator import iso_evaluator_node
from deterministic_tools.machine_expert_nodes import (
    fallback_band_node,
    kurtogram_node,
    kurtogram_reliability_router,
    machine_type_router,
    reciprocating_expert_node,
    rotating_expert_node,
)
from schemas.states import SignalProcessingInternalState, VibraDiagMainState

logger = logging.getLogger(__name__)


def iso_check_node(state: SignalProcessingInternalState) -> dict:
    """ISO değerlendirmesi için gerekli alanlar mevcutsa evaluator'ı çalıştırır."""
    if "vibration_velocity_rms_mm_s" not in state or "machine_class" not in state:
        return {}
    return iso_evaluator_node(state)


def build_signal_processing_subgraph():
    """Tüm sinyal işleme düğümlerini tek bir derlenmiş alt-grafta toplar."""
    graph = StateGraph(SignalProcessingInternalState)

    def _safe_node(fn):
        def _wrapper(s):
            res = fn(s)
            return _sanitize_for_state(res) if res else res
        return _wrapper

    graph.add_node("iso_check", _safe_node(iso_evaluator_node))
    graph.add_node("kurtogram", _safe_node(kurtogram_node))
    graph.add_node("fallback_band", _safe_node(fallback_band_node))
    graph.add_node("rotating_expert", _safe_node(rotating_expert_node))
    graph.add_node("reciprocating_expert", _safe_node(reciprocating_expert_node))

    graph.set_entry_point("iso_check")
    graph.add_conditional_edges(
        "iso_check",
        machine_type_router,
        {"rotating": "kurtogram", "reciprocating": "reciprocating_expert"},
    )
    graph.add_conditional_edges(
        "kurtogram",
        kurtogram_reliability_router,
        {"rotating_expert": "rotating_expert", "fallback_band": "fallback_band"},
    )
    graph.add_edge("fallback_band", "rotating_expert")
    graph.add_edge("rotating_expert", END)
    graph.add_edge("reciprocating_expert", END)

    return graph.compile()


_COMPILED_SUBGRAPH = None


def _get_compiled_subgraph():
    global _COMPILED_SUBGRAPH
    if _COMPILED_SUBGRAPH is None:
        _COMPILED_SUBGRAPH = build_signal_processing_subgraph()
    return _COMPILED_SUBGRAPH


def _rectangularize_kurtogram(kurtogram: list[Any]) -> list[list[float]]:
    """
    Converts a ragged kurtogram decomposition tree into a uniform 2D rectangular matrix
    by repeating values across the highest resolution level width.
    """
    import numpy as np
    if not kurtogram:
        return []

    valid_rows = [row for row in kurtogram if hasattr(row, "__len__") and len(row) > 0]
    if not valid_rows:
        return []

    max_cols = max(len(row) for row in valid_rows)
    rect_matrix = []
    for row in valid_rows:
        row_arr = np.asarray(row, dtype=float)
        n = len(row_arr)
        if n == max_cols:
            rect_matrix.append(row_arr.tolist())
        else:
            rep = max(1, max_cols // n)
            expanded = np.repeat(row_arr, rep)
            if len(expanded) < max_cols:
                expanded = np.pad(expanded, (0, max_cols - len(expanded)), mode="edge")
            elif len(expanded) > max_cols:
                expanded = expanded[:max_cols]
            rect_matrix.append(expanded.tolist())

    return rect_matrix


def plot_diagnostics_node(state: dict) -> dict[str, Any]:
    """
    Generate Plotly figures for FFT spectrum and Kurtogram heatmap,
    storing them as serialized dictionaries in state['diagnostic_plots'].
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy.signal.windows import hann

    plots = {}
    signal_res = state.get("signal_processing_result") or state.get("signal_analysis", {})
    if isinstance(signal_res, object) and hasattr(signal_res, "model_dump"):
        signal_res = signal_res.model_dump()

    freqs = None
    mag = None

    raw_results = signal_res.get("raw_results", {}) if isinstance(signal_res, dict) else {}
    fft_data = raw_results.get("fft", {}) if isinstance(raw_results, dict) else {}
    if isinstance(fft_data, dict) and "freqs" in fft_data and "magnitude" in fft_data:
        freqs = fft_data["freqs"]
        mag = fft_data["magnitude"]

    if freqs is None or mag is None:
        loaded = state.get("loaded_signal") or (signal_res.get("loaded_signal") if isinstance(signal_res, dict) else None)
        if loaded and hasattr(loaded, "channels") and loaded.channels:
            primary_ch = (
                state.get("primary_channel")
                or (signal_res.get("primary_channel") if isinstance(signal_res, dict) else None)
                or ("DE" if "DE" in loaded.channels else next(iter(loaded.channels)))
            )
            signal_data = loaded.channels.get(primary_ch)
            fs = loaded.fs
            if signal_data is not None and len(signal_data) > 0 and fs > 0:
                n = len(signal_data)
                window = hann(n)
                freqs = np.fft.rfftfreq(n, d=1.0 / fs)
                mag = np.abs(np.fft.rfft(signal_data * window))

    if freqs is not None and mag is not None:
        if hasattr(freqs, "tolist"):
            freqs = freqs.tolist()
        if hasattr(mag, "tolist"):
            mag = mag.tolist()

        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs, y=mag, mode="lines", name="FFT Spektrumu", line=dict(color="#1f77b4")))
        fig_fft.update_layout(
            title="Titreşim Sinyali FFT Spektrumu",
            xaxis_title="Frekans (Hz)",
            yaxis_title="Genlik",
            template="plotly_dark",
        )
        plots["fft_spectrum"] = fig_fft.to_dict()

    raw_kurtogram = signal_res.get("kurtogram") if isinstance(signal_res, dict) else None
    if raw_kurtogram:
        rect_kurt = _rectangularize_kurtogram(raw_kurtogram)
        if rect_kurt:
            fig_kurt = go.Figure(data=go.Heatmap(z=rect_kurt, colorscale="Viridis"))
            fig_kurt.update_layout(
                title="Spectral Kurtogram Heatmap",
                xaxis_title="Frekans Bandı Dağılımı",
                yaxis_title="Ayrıştırma Seviyesi (Level 1..N)",
                template="plotly_dark",
            )
            plots["kurtogram"] = fig_kurt.to_dict()

    return {"diagnostic_plots": _sanitize_for_state(plots)}


_SUBGRAPH_INPUT_KEYS = (
    "machine_type",
    "loaded_signal",
    "rpm",
    "n_balls",
    "ball_diameter",
    "pitch_diameter",
    "contact_angle_deg",
    "n_cylinders",
    "stroke_type",
    "machine_class",
    "vibration_velocity_rms_mm_s",
)


def map_parent_to_subgraph(parent_state: VibraDiagMainState) -> SignalProcessingInternalState:
    """VibraDiagMainState -> SignalProcessingInternalState mapping."""
    meta = parent_state.get("machine_metadata") or {}
    subgraph_input: dict[str, Any] = {}
    for k in _SUBGRAPH_INPUT_KEYS:
        if k in parent_state and parent_state[k] is not None:
            subgraph_input[k] = parent_state[k]
        elif k in meta and meta[k] is not None:
            subgraph_input[k] = meta[k]
    return subgraph_input


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)


def _sanitize_for_state(obj: Any) -> Any:
    """Recursively converts all objects, numpy types, and dataclasses to msgpack serializable Python primitives."""
    return json.loads(json.dumps(obj, cls=SafeJSONEncoder))


def map_subgraph_to_parent(subgraph_output: dict) -> dict:
    """SignalProcessingInternalState -> VibraDiagMainState mapping."""
    sanitized = _sanitize_for_state(dict(subgraph_output))
    return {"signal_processing_result": sanitized}


def signal_processing_subgraph_node(parent_state: VibraDiagMainState) -> dict:
    """Ana grafa tek bir node olarak eklenecek wrapper fonksiyon."""
    subgraph_input = map_parent_to_subgraph(parent_state)
    subgraph = _get_compiled_subgraph()
    subgraph_output = subgraph.invoke(subgraph_input)
    return map_subgraph_to_parent(subgraph_output)


def iso_zone_router(parent_state: VibraDiagMainState) -> str:
    """ISO değerlendirmesine göre acil durum yönlendirmesi yapar."""
    result = parent_state.get("signal_processing_result", {})
    zone = result.get("iso_severity_zone")
    return "emergency" if zone == "D" else "continue"
