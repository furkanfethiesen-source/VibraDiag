import base64
import json
import pathlib
import re
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .fault_panel import render_fault_panel

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent


def render_file_chip(filename: str, file_info: str = "") -> None:
    """Renders a file attachment chip in the chat."""
    html = f"""
    <div style="display: inline-block; padding: 4px 12px; background-color: #2D2D2D; border: 1px solid #444; border-radius: 16px; font-size: 0.9em; color: #E0E0E0; margin-bottom: 10px;">
        📎 {filename} <span style="color: #888;">{file_info}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _extract_plotly_from_html(html_str: str) -> go.Figure | None:
    """Extracts Plotly data and layout from standalone HTML script to render natively."""
    try:
        match = re.search(
            r"Plotly\.newPlot\(\s*[\"'][^\"']+[\"']\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*,\s*(\{.*?\})\s*\)",
            html_str,
            re.DOTALL,
        )
        if match:
            d = json.loads(match.group(1))
            l = json.loads(match.group(2))
            return go.Figure(data=d, layout=l)
    except Exception:
        pass
    return None


def _render_plot_element(
    plot_data: Any,
    title: str = "",
    height: int = 240,
    key: str | None = None,
) -> None:
    """Helper to render Plotly figures natively (fitted, no iframe scrollbars) or fallback to HTML."""
    if not plot_data:
        return

    # 1. If it's already a dict (Plotly figure dictionary)
    if isinstance(plot_data, dict):
        try:
            fig = go.Figure(plot_data)
            fig.update_layout(
                height=height,
                margin=dict(l=45, r=20, t=35, b=35),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(25,25,25,0.7)",
                title_font=dict(size=14, color="#E0E0E0"),
                xaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                yaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                autosize=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
            return
        except Exception as e:
            st.warning(f"Grafik çizilemedi: {e}")
            return

    # 2. If it's a string
    if isinstance(plot_data, str):
        cleaned = plot_data.strip()

        # 2a. Check if it's a file path
        p = pathlib.Path(cleaned)
        if not p.is_absolute():
            root_candidate = ROOT_DIR / cleaned
            if root_candidate.exists():
                p = root_candidate

        if p.exists() and p.is_file():
            if p.suffix == ".html":
                try:
                    html_content = p.read_text(encoding="utf-8")
                    fig = _extract_plotly_from_html(html_content)
                    if fig:
                        fig.update_layout(
                            height=height,
                            margin=dict(l=45, r=20, t=35, b=35),
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(25,25,25,0.7)",
                            title_font=dict(size=14, color="#E0E0E0"),
                            xaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                            yaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                            autosize=True,
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
                        return
                    else:
                        components.html(html_content, height=height, scrolling=False)
                        return
                except Exception as e:
                    st.warning(f"HTML dosyası okunamadı ({p.name}): {e}")
                    return
            elif p.suffix == ".json":
                try:
                    with open(p, encoding="utf-8") as f:
                        dict_data = json.load(f)
                    fig = go.Figure(dict_data)
                    fig.update_layout(
                        height=height,
                        margin=dict(l=45, r=20, t=35, b=35),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(25,25,25,0.7)",
                        title_font=dict(size=14, color="#E0E0E0"),
                        xaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                        yaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                        autosize=True,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
                    return
                except Exception as e:
                    st.warning(f"JSON grafik dosyası okunamadı: {e}")
                    return

        # 2b. Check if it's raw HTML
        if cleaned.startswith("<"):
            fig = _extract_plotly_from_html(cleaned)
            if fig:
                fig.update_layout(
                    height=height,
                    margin=dict(l=45, r=20, t=35, b=35),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(25,25,25,0.7)",
                    title_font=dict(size=14, color="#E0E0E0"),
                    xaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                    yaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                    autosize=True,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
                return
            components.html(cleaned, height=height, scrolling=False)
            return

        # 2c. Check if it's a JSON string
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                dict_data = json.loads(cleaned)
                fig = go.Figure(dict_data)
                fig.update_layout(
                    height=height,
                    margin=dict(l=45, r=20, t=35, b=35),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(25,25,25,0.7)",
                    title_font=dict(size=14, color="#E0E0E0"),
                    xaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                    yaxis=dict(title_font=dict(size=12), tickfont=dict(size=11)),
                    autosize=True,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
                return
            except Exception:
                pass

        st.caption(f"📊 {title}: {cleaned}")


def render_diagnostic_card(diagnosis_response: dict, card_index: int | str = 0) -> None:
    """Renders the full diagnostic card inside a chat message."""
    if not isinstance(diagnosis_response, dict):
        return

    st.markdown('<div class="diagnostic-card">', unsafe_allow_html=True)

    col_left, col_right = st.columns([6.8, 3.2])

    plots = (
        diagnosis_response.get("diagnostic_plots")
        or diagnosis_response.get("saved_plots")
        or diagnosis_response.get("plots")
        or {}
    )
    if isinstance(plots, object) and hasattr(plots, "model_dump"):
        plots = plots.model_dump()
    elif not isinstance(plots, dict):
        plots = {}

    fft = (
        plots.get("fft_spectrum_dict")
        or plots.get("fft_spectrum")
        or plots.get("fft_spectrum_html")
        or plots.get("fft_spectrum_path")
    )
    envelope = (
        plots.get("envelope_spectrum_dict")
        or plots.get("envelope_spectrum")
        or plots.get("envelope_spectrum_html")
        or plots.get("envelope_spectrum_path")
    )
    kurtogram = (
        plots.get("kurtogram_dict")
        or plots.get("kurtogram")
        or plots.get("kurtogram_html")
        or plots.get("kurtogram_path")
    )
    orbit = (
        plots.get("orbit_plot_dict")
        or plots.get("orbit_plot")
        or plots.get("orbit_plot_html")
        or plots.get("orbit_plot_path")
    )

    primary_fault_data = diagnosis_response.get("primary_fault_data") or {}
    fault_type_abbr = (
        primary_fault_data.get("fault_type_abbr")
        or primary_fault_data.get("abbr")
        or ""
    ).upper()

    fault_map = {
        "FTF": "bearing_ftf.jpg",
        "BPFO": "bearing_bpfo.jpg",
        "BPFI": "bearing_bpfi.jpg",
        "BSF": "bearing_bsf.jpg",
        "UNBALANCE": "fault_unbalance.jpg",
        "MISALIGNMENT": "fault_misalignment.jpg",
        "LOOSENESS": "fault_looseness.jpg",
    }
    filename = fault_map.get(fault_type_abbr, "bearing_bpfo.jpg")
    asset_path = pathlib.Path(__file__).parent.parent / "assets" / filename

    card_id = (
        diagnosis_response.get("signal_id")
        or diagnosis_response.get("session_id")
        or f"{card_index}_{id(diagnosis_response)}"
    )
    fft_key = f"fft_chart_{card_id}_{card_index}"
    env_key = f"env_chart_{card_id}_{card_index}"
    kurt_key = f"kurt_chart_{card_id}_{card_index}"
    orbit_key = f"orbit_chart_{card_id}_{card_index}"

    with col_left:
        # Row 1: FFT Spectrum (Full width of the left area)
        st.markdown("### 📈 FFT Spektrumu")
        if fft:
            _render_plot_element(fft, title="FFT Spektrumu", height=240, key=fft_key)
        else:
            st.info("FFT verisi mevcut değil.")

        st.markdown('<div style="margin-top: 6px;"></div>', unsafe_allow_html=True)

        # Row 2: Envelope Spectrum (Full width of the left area)
        st.markdown("### 📊 Zarf Spektrumu (Envelope Spectrum)")
        if envelope:
            _render_plot_element(envelope, title="Zarf Spektrumu", height=240, key=env_key)
        else:
            st.info("Zarf spektrumu mevcut değil.")

        st.markdown('<div style="margin-top: 6px;"></div>', unsafe_allow_html=True)

        # Row 3: Kurtogram Heatmap (Full width of the left area)
        st.markdown("### 🔥 Kurtogram Heatmap")
        if kurtogram:
            _render_plot_element(kurtogram, title="Spectral Kurtogram Heatmap", height=220, key=kurt_key)
        else:
            st.info("Kurtogram ısı haritası mevcut değil.")

        st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)

        # Row 4: Polar Orbit Plot & Fault Location 3D Asset side-by-side [1, 1]
        row4_col1, row4_col2 = st.columns([1, 1])
        with row4_col1:
            st.markdown("### 🔄 Mil Yörüngesi (Orbit)")
            if orbit:
                _render_plot_element(orbit, title="Mil Yörüngesi (Orbit Plot)", height=280, key=orbit_key)
            else:
                st.info("Yörünge verisi mevcut değil.")

        with row4_col2:
            st.markdown("### ⚙️ Fault Location")
            if asset_path.exists():
                try:
                    img_bytes = asset_path.read_bytes()
                    b64_img = base64.b64encode(img_bytes).decode("utf-8")
                    card_html = f"""
                    <div style="height: 280px; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: rgba(25,25,25,0.7); border-radius: 8px; border: 1px solid #333; padding: 12px; box-sizing: border-box;">
                        <img src="data:image/jpeg;base64,{b64_img}" style="max-height: 215px; max-width: 100%; object-fit: contain; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);" />
                        <div style="font-size: 0.9em; color: #BBB; margin-top: 8px; font-weight: 500;">Teşhis: <strong style="color: #D4AF37;">{fault_type_abbr or 'Rulman'}</strong></div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                except Exception:
                    st.image(str(asset_path), use_container_width=True, caption=f"Teşhis: {fault_type_abbr or 'Rulman'}")
            else:
                st.info(f"Görsel bulunamadı: {filename}")

    with col_right:
        render_fault_panel(diagnosis_response)

    st.markdown("</div>", unsafe_allow_html=True)
