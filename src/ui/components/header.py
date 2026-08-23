import datetime

import streamlit as st


def render_header(
    system_status: str = "OK",
    sensor_id: str = "—",
    timestamp: str = "",
    machine_class: str | None = None,
    severity_zone: str | None = None,
) -> None:
    """Renders the top status bar of the application."""
    if not timestamp:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    dot_color = "#00FF00" if system_status.upper() == "OK" else "#FF0000"

    badges = ""
    if machine_class:
        badges += f'<span class="severity-badge" style="padding: 2px 8px; border-radius: 4px; background-color: #333; color: #D4AF37; border: 1px solid #D4AF37; font-size: 0.85em;">{machine_class}</span>'

    if severity_zone:
        zone_class = severity_zone.lower()
        color_map = {"a": "#00FF00", "b": "#0000FF", "c": "#FFC000", "d": "#FF0000"}
        zone_color = color_map.get(zone_class, "#777")
        badges += f'<span class="severity-badge severity-zone-{zone_class}" style="padding: 2px 8px; border-radius: 4px; background-color: {zone_color}22; color: {zone_color}; border: 1px solid {zone_color}; font-size: 0.85em;">Zone {severity_zone.upper()}</span>'

    html = (
        f'<div class="header-bar" style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: #1E1E1E; border-bottom: 1px solid #333;">'
        f'<div style="display: flex; align-items: center; gap: 10px; color: #E0E0E0;">'
        f'<div class="status-dot" style="width: 10px; height: 10px; border-radius: 50%; background-color: {dot_color}; box-shadow: 0 0 5px {dot_color};"></div>'
        f'<strong style="color: #D4AF37;">VibraDiag-AI</strong>'
        f'<span style="color: #666;">|</span>'
        f'<span>{system_status}</span>'
        f'<span style="color: #666;">|</span>'
        f'<span>{sensor_id}</span>'
        f'<span style="color: #666;">|</span>'
        f'<span style="font-size: 0.9em; color: #AAA;">{timestamp}</span>'
        f'</div>'
        f'<div style="display: flex; align-items: center; gap: 10px;">{badges}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
