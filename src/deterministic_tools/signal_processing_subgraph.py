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
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from scipy.signal.windows import hann
from scipy.signal import hilbert, butter, sosfilt

from config_loader import load_appcfg

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
    machine_class = state.get("machine_class")
    if not machine_class:
        return {}

    rms = state.get("vibration_velocity_rms_mm_s")
    if rms is None:
        loaded = state.get("loaded_signal")
        if loaded and hasattr(loaded, "channels") and loaded.channels:
            import numpy as np
            primary_ch = state.get("primary_channel") or ("DE" if "DE" in loaded.channels else next(iter(loaded.channels)))
            sig = loaded.channels.get(primary_ch)
            if sig is not None and len(sig) > 0:
                rms = float(np.sqrt(np.mean(sig**2)))

    if rms is None:
        return {}

    eval_state = dict(state)
    eval_state["vibration_velocity_rms_mm_s"] = rms
    res = iso_evaluator_node(eval_state)
    res["vibration_velocity_rms_mm_s"] = rms
    return res


def build_signal_processing_subgraph():
    """Tüm sinyal işleme düğümlerini tek bir derlenmiş alt-grafta toplar."""
    graph = StateGraph(SignalProcessingInternalState)

    def _safe_node(fn):
        def _wrapper(s):
            res = fn(s)
            return _sanitize_for_state(res) if res else res
        return _wrapper

    graph.add_node("iso_check", _safe_node(iso_check_node))
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


def _rectangularize_kurtogram(kurtogram: list[Any]) -> tuple[list[list[float]], list[str]]:
    """
    Converts a ragged kurtogram decomposition tree into a uniform 2D rectangular matrix
    by repeating values across the highest resolution level width, and returns 1-based level labels.
    """
    import numpy as np
    if not kurtogram:
        return [], []

    valid_rows = [row for row in kurtogram if hasattr(row, "__len__") and len(row) > 0]
    if not valid_rows:
        return [], []

    max_cols = max(len(row) for row in valid_rows)
    rect_matrix = []
    level_labels = [f"Level {i+1}" for i in range(len(valid_rows))]
    for row in valid_rows:
        row_arr = np.asarray(row, dtype=float)
        row_arr = np.nan_to_num(row_arr, nan=0.0, neginf=0.0, posinf=0.0)
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

    return rect_matrix, level_labels


def plot_diagnostics_node(state: dict) -> dict[str, Any]:
    """
    Generate Plotly figures for FFT spectrum, Envelope spectrum, and Kurtogram heatmap,
    save them as standalone interactive HTML files on disk, and store
    lightweight file paths in state['diagnostic_plots'].
    """
    from deterministic_tools.signal_processing import bandpass_filter, hilbert_envelope, envelope_fft, calc_fault_freqs

    app_cfg = load_appcfg()
    paths_cfg = getattr(app_cfg, "paths", {}) or {}
    output_dir_str = paths_cfg.get("sample_outputs_dir", "sample_outputs")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    import time
    session_id = state.get("session_id") or state.get("route_decision") or "signal_only"
    signal_file = state.get("signal_file_path") or ""
    signal_stem = Path(str(signal_file)).stem if signal_file else ""
    ts_suffix = str(int(time.time() * 1000))[-6:]
    base_prefix = f"{session_id}_{signal_stem}_{ts_suffix}" if signal_stem else f"{session_id}_{ts_suffix}"
    clean_prefix = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_prefix)

    plots = {}
    signal_res = state.get("signal_processing_result") or state.get("signal_analysis", {})
    if isinstance(signal_res, object) and hasattr(signal_res, "model_dump"):
        signal_res = signal_res.model_dump()

    loaded = state.get("loaded_signal") or (signal_res.get("loaded_signal") if isinstance(signal_res, dict) else None)
    if not loaded:
        signal_file = state.get("signal_file_path")
        if signal_file and Path(str(signal_file)).exists():
            from deterministic_tools.reader import SignalReaderFactory
            meta = state.get("machine_metadata") or {}
            expected_fs = meta.get("fs") or meta.get("expected_fs")
            try:
                loaded = SignalReaderFactory.load(str(signal_file), expected_fs=expected_fs)
            except Exception as e:
                logger.warning("plot_diagnostics_node could not load signal '%s': %s", signal_file, e)

    primary_ch = (
        state.get("primary_channel")
        or (signal_res.get("primary_channel") if isinstance(signal_res, dict) else None)
        or ("DE" if loaded and hasattr(loaded, "channels") and "DE" in loaded.channels else (next(iter(loaded.channels)) if loaded and hasattr(loaded, "channels") and loaded.channels else "DE"))
    )

    signal_data = None
    fs = 12000.0
    if loaded and hasattr(loaded, "channels") and loaded.channels:
        if primary_ch in loaded.channels:
            signal_data = loaded.channels[primary_ch]
        elif loaded.channels:
            signal_data = next(iter(loaded.channels.values()))
        fs = float(loaded.fs)

    # 1. FFT Spectrum
    fft_freqs = None
    fft_mag = None
    raw_results = signal_res.get("raw_results", {}) if isinstance(signal_res, dict) else {}
    fft_data = raw_results.get("fft", {}) if isinstance(raw_results, dict) else {}
    if isinstance(fft_data, dict) and "freqs" in fft_data and "magnitude" in fft_data:
        fft_freqs = np.asarray(fft_data["freqs"])
        fft_mag = np.asarray(fft_data["magnitude"])
    elif signal_data is not None and len(signal_data) > 0 and fs > 0:
        n = len(signal_data)
        window = hann(n)
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        fft_mag = np.abs(np.fft.rfft(signal_data * window)) / n * 2.0

    if fft_freqs is not None and fft_mag is not None:
        if len(fft_freqs) > 4000:
            step = int(np.ceil(len(fft_freqs) / 4000))
            fft_freqs = fft_freqs[::step]
            fft_mag = fft_mag[::step]

        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=fft_freqs.tolist(),
            y=fft_mag.tolist(),
            mode="lines",
            name="FFT",
            line=dict(color="#29B6F6", width=1.5),
        ))
        fig_fft.update_layout(
            title=dict(text="Titreşim Sinyali FFT Spektrumu", font=dict(color="#E0E0E0", size=14)),
            xaxis=dict(title=dict(text="Frekans (Hz)", font=dict(size=12)), tickfont=dict(size=11), gridcolor="#333", zeroline=False),
            yaxis=dict(title=dict(text="Genlik", font=dict(size=12)), tickfont=dict(size=11), gridcolor="#333", zeroline=False),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(25,25,25,0.7)",
            margin=dict(l=45, r=20, t=35, b=35),
            autosize=True,
        )
        fft_path = output_dir / f"{clean_prefix}_fft_spectrum.html"
        fig_fft.write_html(str(fft_path), include_plotlyjs="cdn")
        fft_html_path_str = str(fft_path)
        plots["fft_spectrum"] = fft_html_path_str
        plots["fft_spectrum_path"] = fft_html_path_str
        plots["fft_spectrum_dict"] = fig_fft.to_dict()

    # 2. Envelope Spectrum
    if signal_data is not None and len(signal_data) > 0 and fs > 0:
        band = state.get("final_band") or state.get("kurtogram_band")
        if not band and isinstance(signal_res, dict):
            band = signal_res.get("final_band") or signal_res.get("kurtogram_band")
        if not band:
            band = (max(500.0, fs * 0.1), min(fs * 0.45, 5000.0))

        try:
            filt = bandpass_filter(signal_data, fs, band)
            env = hilbert_envelope(filt)
            env_freqs, env_mag = envelope_fft(env, fs, remove_dc=True, high_pass_cutoff_hz=2.0)

            # Limit to 0-500 Hz for bearing envelope diagnostics
            mask_500 = (env_freqs >= 0.0) & (env_freqs <= min(500.0, fs / 2.0))
            ef_plot = env_freqs[mask_500]
            em_plot = env_mag[mask_500]

            fig_env = go.Figure()
            fig_env.add_trace(go.Scatter(
                x=ef_plot.tolist(),
                y=em_plot.tolist(),
                mode="lines",
                name="Zarf Spektrumu",
                line=dict(color="#FFB300", width=1.6),
            ))

            # Add vertical lines for characteristic bearing harmonics if metadata is present
            meta = state.get("machine_metadata") or {}
            rpm = float(state.get("rpm") or meta.get("rpm") or 1797.0)
            n_balls = int(state.get("n_balls") or meta.get("n_balls") or 9)
            ball_diam = float(state.get("ball_diameter") or meta.get("ball_diameter") or 0.3126)
            pitch_diam = float(state.get("pitch_diameter") or meta.get("pitch_diameter") or 1.537)
            contact_deg = float(state.get("contact_angle_deg") or meta.get("contact_angle_deg") or 0.0)

            ff = calc_fault_freqs(rpm=rpm, n_balls=n_balls, ball_diameter=ball_diam, pitch_diameter=pitch_diam, contact_angle_deg=contact_deg)
            
            primary_data = signal_res.get("primary_fault_data") or state.get("primary_fault_data") or {}
            abbr = str(primary_data.get("fault_type_abbr") or "BPFO").upper()
            target_base = ff.get(abbr) or ff.get("BPFO") or 107.4
            
            # Draw 1X, 2X, 3X dashed marker lines
            max_plot_y = float(np.max(em_plot)) if len(em_plot) > 0 else 1.0
            for harm in [1, 2, 3]:
                h_freq = target_base * harm
                if h_freq <= 500.0:
                    fig_env.add_vline(
                        x=h_freq,
                        line_dash="dash",
                        line_color="#E040FB",
                        annotation_text=f"{harm}X {abbr} ({h_freq:.1f}Hz)",
                        annotation_position="top right",
                        annotation_font=dict(color="#E040FB", size=11),
                    )

            fig_env.update_layout(
                title=dict(text=f"📊 Zarf Spektrumu ({abbr} Harmonikleri)", font=dict(color="#E0E0E0", size=14)),
                xaxis=dict(title=dict(text="Frekans (Hz)", font=dict(size=12)), tickfont=dict(size=11), range=[0, min(500.0, fs / 2.0)], gridcolor="#333", zeroline=False),
                yaxis=dict(title=dict(text="Zarf Genliği", font=dict(size=12)), tickfont=dict(size=11), gridcolor="#333", zeroline=False),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(25,25,25,0.7)",
                margin=dict(l=45, r=20, t=35, b=35),
                autosize=True,
            )
            env_path = output_dir / f"{clean_prefix}_envelope_spectrum.html"
            fig_env.write_html(str(env_path), include_plotlyjs="cdn")
            plots["envelope_spectrum"] = str(env_path)
            plots["envelope_spectrum_path"] = str(env_path)
            plots["envelope_spectrum_dict"] = fig_env.to_dict()
        except Exception as e:
            logger.warning("Could not generate envelope plot: %s", e)

    # 3. Kurtogram Heatmap
    raw_kurtogram = state.get("kurtogram") or (signal_res.get("kurtogram") if isinstance(signal_res, dict) else None)
    if raw_kurtogram:
        rect_kurt, y_labels = _rectangularize_kurtogram(raw_kurtogram)
        if rect_kurt:
            fig_kurt = go.Figure(data=go.Heatmap(
                z=rect_kurt,
                y=y_labels if y_labels else None,
                colorscale="Viridis",
                colorbar=dict(title=dict(text="SK", font=dict(size=12)), tickfont=dict(size=10)),
            ))
            fig_kurt.update_layout(
                title=dict(text="🔥 Spektral Kurtogram Isı Haritası", font=dict(color="#E0E0E0", size=14)),
                xaxis=dict(title=dict(text="Frekans Dağılımı (Bantlar)", font=dict(size=12)), tickfont=dict(size=11), gridcolor="#333"),
                yaxis=dict(title=dict(text="Ayrıştırma Seviyesi", font=dict(size=12)), tickfont=dict(size=11), gridcolor="#333"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(25,25,25,0.7)",
                margin=dict(l=45, r=20, t=35, b=35),
                autosize=True,
            )
            kurt_path = output_dir / f"{clean_prefix}_kurtogram.html"
            fig_kurt.write_html(str(kurt_path), include_plotlyjs="cdn")
            plots["kurtogram"] = str(kurt_path)
            plots["kurtogram_path"] = str(kurt_path)
            plots["kurtogram_dict"] = fig_kurt.to_dict()

    # 4. Polar Orbit / Phase Portrait Plot
    if signal_data is not None and len(signal_data) > 0 and fs > 0:
        try:
            ch_names = list(loaded.channels.keys()) if (loaded and hasattr(loaded, "channels") and loaded.channels) else []
            is_orthogonal = False
            x_raw, y_raw = None, None

            ortho_pairs = [
                ("X", "Y"), ("RADIAL_H", "RADIAL_V"), ("CH1", "CH2"),
                ("DE_H", "DE_V"), ("FE_H", "FE_V"), ("HORIZ", "VERT")
            ]
            upper_map = {name.upper(): name for name in ch_names}
            for p1, p2 in ortho_pairs:
                if p1 in upper_map and p2 in upper_map:
                    x_raw = loaded.channels[upper_map[p1]]
                    y_raw = loaded.channels[upper_map[p2]]
                    is_orthogonal = True
                    break

            f_rot = float(rpm) / 60.0 if rpm and rpm > 0 else 29.95

            lowcut = max(2.0, f_rot * 0.4)
            highcut = min(fs * 0.45, max(lowcut + 10.0, f_rot * 3.5))
            sos = butter(4, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")

            if is_orthogonal and x_raw is not None and y_raw is not None:
                min_len = min(len(x_raw), len(y_raw))
                x_filt = sosfilt(sos, x_raw[:min_len])
                y_filt = sosfilt(sos, y_raw[:min_len])
                orbit_title = "🔄 Mil Yörüngesi (Orbit Plot - X/Y)"
                x_label = "X Ekseni (Deplasman)"
                y_label = "Y Ekseni (Deplasman)"
            else:
                x_filt = sosfilt(sos, signal_data)
                analytic_sig = hilbert(x_filt)
                y_filt = np.imag(analytic_sig)
                orbit_title = "🔄 Mil Yörüngesi (Faz Portresi)"
                x_label = "Faz Bileşeni (x)"
                y_label = "Kuadratür Bileşeni (x̂)"

            pts_per_rev = int(round(fs / f_rot)) if f_rot > 0 else 400
            n_revs = 6
            total_pts = min(len(x_filt), pts_per_rev * n_revs)
            start_idx = max(0, min(pts_per_rev * 2, len(x_filt) - total_pts))

            x_plot = x_filt[start_idx : start_idx + total_pts]
            y_plot = y_filt[start_idx : start_idx + total_pts]

            if len(x_plot) > 1200:
                step = int(np.ceil(len(x_plot) / 1200))
                x_plot = x_plot[::step]
                y_plot = y_plot[::step]

            fig_orbit = go.Figure()
            fig_orbit.add_trace(go.Scatter(
                x=x_plot.tolist(),
                y=y_plot.tolist(),
                mode="lines",
                name="Yörünge",
                line=dict(color="#00E5FF", width=2.0),
            ))
            fig_orbit.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                name="Mil Merkezi",
                marker=dict(size=8, color="#D4AF37", symbol="cross"),
                showlegend=False,
            ))

            fig_orbit.update_layout(
                title=dict(text=orbit_title, font=dict(color="#E0E0E0", size=14)),
                xaxis=dict(
                    title=dict(text=x_label, font=dict(size=12)),
                    tickfont=dict(size=11),
                    gridcolor="#333",
                    zeroline=True,
                    zerolinecolor="#555",
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(size=12)),
                    tickfont=dict(size=11),
                    gridcolor="#333",
                    zeroline=True,
                    zerolinecolor="#555",
                    scaleanchor="x",
                    scaleratio=1,
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(25,25,25,0.7)",
                margin=dict(l=45, r=20, t=35, b=35),
                autosize=True,
            )

            orbit_path = output_dir / f"{clean_prefix}_orbit_plot.html"
            fig_orbit.write_html(str(orbit_path), include_plotlyjs="cdn")
            plots["orbit_plot"] = str(orbit_path)
            plots["orbit_plot_path"] = str(orbit_path)
            plots["orbit_plot_dict"] = fig_orbit.to_dict()
        except Exception as e:
            logger.warning("Could not generate orbit plot: %s", e)

    return {"diagnostic_plots": _sanitize_for_state(plots)}


_SUBGRAPH_INPUT_KEYS = (
    "machine_type",
    "loaded_signal",
    "signal_file_path",
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
    raw_dict = dict(subgraph_output)
    raw_dict.pop("loaded_signal", None)
    raw_dict.pop("kurtogram", None)
    sanitized = _sanitize_for_state(raw_dict)
    return {"signal_processing_result": sanitized}


def signal_processing_subgraph_node(parent_state: VibraDiagMainState) -> dict:
    """Ana grafa tek bir node olarak eklenecek wrapper fonksiyon."""
    subgraph_input = map_parent_to_subgraph(parent_state)
    if "loaded_signal" not in subgraph_input or subgraph_input["loaded_signal"] is None:
        signal_file = parent_state.get("signal_file_path")
        if signal_file:
            from deterministic_tools.reader import SignalReaderFactory
            meta = parent_state.get("machine_metadata") or {}
            expected_fs = meta.get("fs") or meta.get("expected_fs")
            subgraph_input["loaded_signal"] = SignalReaderFactory.load(str(signal_file), expected_fs=expected_fs)

    state = dict(subgraph_input)
    iso_res = iso_check_node(state)
    state.update(iso_res)

    mtype = machine_type_router(state)
    if mtype == "rotating":
        kurt_res = kurtogram_node(state)
        state.update(kurt_res)

        if not state.get("kurtogram_reliable"):
            fallback_res = fallback_band_node(state)
            state.update(fallback_res)

        rot_res = rotating_expert_node(state)
        state.update(rot_res)
    else:
        recip_res = reciprocating_expert_node(state)
        state.update(recip_res)

    from deterministic_tools.fault_analyzer import pick_primary_fault
    fault_summary = pick_primary_fault(
        direct_spectrum_diagnosis=state.get("direct_spectrum_diagnosis"),
        envelope_diagnosis=state.get("envelope_diagnosis"),
        reciprocating_diagnosis=state.get("reciprocating_diagnosis"),
    )
    state["primary_fault_data"] = fault_summary
    state["primary_fault"] = fault_summary.get("primary_fault_name")
    state["primary_fault_abbr"] = fault_summary.get("fault_type_abbr")

    plots_res = plot_diagnostics_node(state)

    dsp_output = map_subgraph_to_parent(state)
    dsp_output["primary_fault"] = fault_summary.get("primary_fault_name")
    dsp_output["primary_fault_data"] = fault_summary
    dsp_output.update(plots_res)
    return dsp_output


def iso_zone_router(parent_state: VibraDiagMainState) -> str:
    """ISO değerlendirmesine göre acil durum yönlendirmesi yapar."""
    result = parent_state.get("signal_processing_result", {})
    zone = result.get("iso_severity_zone")
    return "emergency" if zone == "D" else "continue"
