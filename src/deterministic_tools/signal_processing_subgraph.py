"""
VibraDiag - Signal Processing SubGraph ve İki Seviyeli State Katmanı
=====================================================================

Bu modül, deterministik sinyal işleme adımlarını (kurtogram, fallback band,
rotating/reciprocating expert, ISO değerlendirmesi) tek bir derlenmiş alt-grafta toplar.

Global state (VibraDiagMainState) ve local subgraph state (SignalProcessingInternalState)
tanımları `schemas.states` modülünden alınır.
"""

from __future__ import annotations

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

    graph.add_node("iso_check", iso_check_node)
    graph.add_node("kurtogram", kurtogram_node)
    graph.add_node("fallback_band", fallback_band_node)
    graph.add_node("rotating_expert", rotating_expert_node)
    graph.add_node("reciprocating_expert", reciprocating_expert_node)

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

    # 1. FFT Spectrum Plot
    freqs = None
    mag = None

    # Check for direct pre-computed fft in raw_results (fallback/mock)
    raw_results = signal_res.get("raw_results", {}) if isinstance(signal_res, dict) else {}
    fft_data = raw_results.get("fft", {}) if isinstance(raw_results, dict) else {}
    if isinstance(fft_data, dict) and "freqs" in fft_data and "magnitude" in fft_data:
        freqs = fft_data["freqs"]
        mag = fft_data["magnitude"]

    # Compute FFT directly from loaded_signal if not pre-computed
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

    # 2. Kurtogram Heatmap Plot
    kurtogram = state.get("kurtogram") or (signal_res.get("kurtogram") if isinstance(signal_res, dict) else None)
    if kurtogram:
        fig_kurt = go.Figure(data=go.Heatmap(z=kurtogram, colorscale="Viridis"))
        fig_kurt.update_layout(
            title="Spectral Kurtogram Heatmap",
            xaxis_title="Frekans Bandı İndeksi",
            yaxis_title="Ayrıştırma Seviyesi (Level)",
            template="plotly_dark",
        )
        plots["kurtogram"] = fig_kurt.to_dict()

    return {"diagnostic_plots": plots}


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
    return {k: parent_state[k] for k in _SUBGRAPH_INPUT_KEYS if k in parent_state}


def map_subgraph_to_parent(subgraph_output: dict) -> dict:
    """SignalProcessingInternalState -> VibraDiagMainState mapping."""
    return {"signal_processing_result": dict(subgraph_output)}


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
