import streamlit as st


def render_fault_panel(diagnosis_response: dict) -> None:
    """Renders the fault diagnosis metrics panel with multi-channel and severity hierarchy."""

    fault_data = diagnosis_response.get("primary_fault_data", {})
    fault_type_abbr = (
        fault_data.get("fault_type_abbr")
        or fault_data.get("abbr")
        or "UNKNOWN"
    ).upper()
    primary_fault_name = fault_data.get("primary_fault_name") or fault_data.get("name", "")
    confidence = float(fault_data.get("confidence", 0.0) or 0.0)
    primary_severity = float(fault_data.get("severity", 0.0) or 0.0)
    dominant_channel = fault_data.get("dominant_channel") or ""

    if confidence > 0.8:
        conf_color = "#D4AF37"
    elif confidence >= 0.5:
        conf_color = "#FFC000"
    else:
        conf_color = "#FF4B4B"

    channel_name_map = {
        "DE": "Tahrik Ucu (Drive End)",
        "FE": "Fan Ucu (Fan End)",
        "BA": "Gövde Tabanı (Base)",
    }
    dominant_label = channel_name_map.get(dominant_channel, dominant_channel)

    # 1. Header Card (Primary Fault + Dominant Source + Severity)
    source_badge = f'<div style="display: inline-block; margin-top: 5px; padding: 2px 8px; background-color: #333; border: 1px solid #D4AF37; border-radius: 4px; font-size: 0.78em; color: #D4AF37;">🎯 Ana Kaynak: {dominant_label}</div>' if dominant_label else ''
    sev_badge = f'<div style="color: #AAA; font-size: 0.85em; margin-top: 3px;">Arıza Şiddeti (Severity): <strong style="color: #FFF;">{primary_severity:.1f}</strong></div>' if primary_severity > 0 else ''

    st.markdown(
        f"""
        <div style="margin-bottom: 12px; padding: 10px; background-color: #262626; border-radius: 8px; border-left: 4px solid {conf_color};">
            <h2 style="margin: 0; color: #E0E0E0; font-size: 1.5em;">{fault_type_abbr}</h2>
            <div style="color: #BBB; font-size: 0.85em; margin-top: 2px;">{primary_fault_name}</div>
            <div style="font-size: 2.0em; font-weight: bold; color: {conf_color}; line-height: 1.1; margin-top: 4px;">{confidence * 100:.0f}% <span style="font-size: 0.45em; color: #888; font-weight: normal;">Güven</span></div>
            {sev_badge}
            {source_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_faults = fault_data.get("all_faults", [])
    bearing_faults = [f for f in all_faults if f.get("category") == "envelope"]
    mech_faults = [f for f in all_faults if f.get("category") in ("direct_spectrum", "reciprocating")]

    # 2. Section 1: Bearing Faults (Envelope Spectrum)
    st.markdown("### 🎯 Rulman Frekans Eşleşmesi")
    st.caption("Zarf spektrumunda teorik harmonik (1X, 2X, 3X) eşleşme oranı")

    has_multichannel = len({f.get("channel") for f in bearing_faults if f.get("channel")}) > 1

    if bearing_faults:
        for idx, fault in enumerate(bearing_faults):
            abbr = (fault.get("abbr") or fault.get("fault_type_abbr") or "").upper()
            channel = fault.get("channel", "")
            name = fault.get("name", "")
            conf = float(fault.get("confidence", 0.0) or 0.0)
            sev = float(fault.get("severity", 0.0) or 0.0)
            pct = int(round(conf * 100))
            n_harm = fault.get("n_harmonics_matched")

            is_dominant = (idx == 0) or (channel and channel == dominant_channel and abbr == fault_type_abbr)

            label_parts = [abbr] if abbr else []
            if channel:
                channel_desc = " (Ana)" if is_dominant else " (İletim)"
                label_parts.append(f"[{channel}{channel_desc}]")
            label_str = " ".join(label_parts) if label_parts else name

            harm_str = f" • {n_harm}/3 harm" if n_harm is not None else ""
            sev_str = f" • Şiddet: {sev:.1f}" if sev > 0 else ""

            bg = "linear-gradient(90deg, #C9A84C, #D4AF37)" if is_dominant else "#4A5568"
            label_color = "#D4AF37" if is_dominant else "#C0C0C0"

            bar_html = f"""
            <div style="margin-bottom: 6px;">
              <div style="display: flex; justify-content: space-between; font-size: 0.82em; color: {label_color}; margin-bottom: 2px;">
                <span><strong>{label_str}</strong><span style="color: #888; font-size: 0.9em;">{harm_str}{sev_str}</span></span>
                <span style="font-weight: bold;">{pct}%</span>
              </div>
              <div style="width: 100%; background-color: #333; border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; border-radius: 4px; background: {bg};"></div>
              </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        if has_multichannel:
            st.info(
                "💡 **Sensör Yayılım Notu:** Titreşim rijit şaft ve gövde boyunca yayıldığı için FE ve BA sensörlerinde de aynı frekans (%100) yakalanmaktadır. En yüksek şiddet ana kaynak sensöründedir.",
                icon="ℹ️",
            )
    else:
        st.caption("Rulman frekans eşleşme detayı bulunamadı.")

    # 3. Section 2: Mechanical & Shaft State (Direct Spectrum)
    if mech_faults:
        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Mekanik & Şaft Durumu")
        st.caption("Doğrudan FFT spektrumu (1X/2X şaft harmonik oranları)")

        for fault in mech_faults:
            abbr = (fault.get("abbr") or fault.get("fault_type_abbr") or "").upper()
            name = fault.get("name", "")
            conf = float(fault.get("confidence", 0.0) or 0.0)
            sev = float(fault.get("severity", 0.0) or 0.0)
            pct = int(round(conf * 100))

            is_minor = (primary_severity > 20.0 and sev < (primary_severity * 0.10))
            minor_badge = " • <span style='color: #888; font-size: 0.85em;'>Düşük Şiddet</span>" if is_minor else ""

            bg = "#4A5568" if not is_minor else "#333333"
            label_color = "#E0E0E0" if not is_minor else "#888888"

            bar_html = f"""
            <div style="margin-bottom: 6px;">
              <div style="display: flex; justify-content: space-between; font-size: 0.82em; color: {label_color}; margin-bottom: 2px;">
                <span><strong>{abbr}</strong> <span style="color: #888; font-size: 0.85em;">(Şiddet: {sev:.1f}){minor_badge}</span></span>
                <span style="font-weight: bold;">{pct}%</span>
              </div>
              <div style="width: 100%; background-color: #333; border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; border-radius: 4px; background: {bg};"></div>
              </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    st.markdown("### ⏱️ Time Domain Stats")

    td_stats = diagnosis_response.get("time_domain_stats", {})
    if isinstance(td_stats, object) and hasattr(td_stats, "model_dump"):
        td_stats = td_stats.model_dump()
    elif not isinstance(td_stats, dict):
        td_stats = {}

    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMS", f"{float(td_stats.get('rms', 0) or td_stats.get('overall_rms', 0) or 0):.2f}")
        st.metric("Crest Factor", f"{float(td_stats.get('crest_factor', 0) or 0):.2f}")
    with col2:
        st.metric("Peak", f"{float(td_stats.get('peak', 0) or td_stats.get('peak_value', 0) or 0):.2f}")
        st.metric("Kurtosis", f"{float(td_stats.get('kurtosis', 0) or 0):.2f}")

    st.markdown("---")

    sidebands = bool(fault_data.get("sidebands_detected", False))
    sideband_icon = "✅ Var" if sidebands else "❌ Yok"
    st.markdown(f"**Yan Bantlar (Sidebands):** {sideband_icon}")

    n_harmonics = fault_data.get("n_harmonics_matched")
    if n_harmonics is None:
        details = fault_data.get("details", {})
        n_harmonics = len(details.get("matched_harmonics", [])) if isinstance(details, dict) else 0
    st.markdown(f"**Eşleşen Harmonikler:** {n_harmonics} / 3")
