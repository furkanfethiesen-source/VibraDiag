import random
import string

import streamlit as st


def render_sidebar(api_base_url: str = "http://localhost:8000") -> dict:
    """Returns a dict with all sidebar parameters and uploaded file info."""
    st.sidebar.markdown('<h1 style="color: #D4AF37;">⚡ VibraDiag</h1>', unsafe_allow_html=True)

    st.sidebar.header("Session Management")

    if not st.session_state.get("session_id"):
        st.session_state["session_id"] = "sess_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

    session_id_val = st.session_state["session_id"]
    entered_id = st.sidebar.text_input("Session ID", value=session_id_val, key="session_id_input")

    col_go, col_new = st.sidebar.columns([1, 1])
    with col_go:
        if st.button("🔍 Oturuma Git", use_container_width=True, help="Yazılan Session ID'ye geçiş yap"):
            if entered_id.strip() and entered_id.strip() != session_id_val:
                st.session_state["session_id"] = entered_id.strip()
                st.session_state.messages = []
                st.session_state.uploaded_signal_id = None
                st.rerun()

    with col_new:
        if st.button("➕ Yeni", use_container_width=True, help="Yeni bir teşhis oturumu başlat"):
            st.session_state["session_id"] = "sess_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
            st.session_state.messages = []
            st.session_state.uploaded_signal_id = None
            st.rerun()

    # Son 5 Oturum Listesi (Recent Chats)
    st.sidebar.markdown('<div style="font-size: 0.85em; font-weight: 600; color: #BBB; margin-top: 10px; margin-bottom: 4px;">🕒 SON OTURUMLAR</div>', unsafe_allow_html=True)
    try:
        from ..api_client import get_client
        client = get_client(api_base_url)
        recent_sessions = client.list_recent_sessions(limit=5)
    except Exception:
        recent_sessions = []

    if recent_sessions:
        for s in recent_sessions:
            s_id = s.get("session_id", "")
            if not s_id:
                continue
            is_active = (s_id == st.session_state["session_id"])
            msg_count = s.get("message_count", 0)
            preview = s.get("last_query_preview") or s.get("primary_fault") or ""
            preview_short = f" • {preview[:18]}..." if preview else ""

            btn_label = f"{'▶️' if is_active else '💬'} {s_id} ({msg_count} msgs{preview_short})"
            btn_key = f"btn_sess_{s_id}"

            if st.sidebar.button(
                btn_label,
                key=btn_key,
                use_container_width=True,
                disabled=is_active,
                help=f"Oturuma Git: {s_id}\nMesaj Sayısı: {msg_count}\nSon Soru/Arıza: {preview or 'N/A'}",
            ):
                st.session_state["session_id"] = s_id
                st.session_state.messages = []
                st.session_state.uploaded_signal_id = None
                st.rerun()
    else:
        st.sidebar.caption("Henüz kayıtlı geçmiş oturum yok.")

    st.sidebar.header("Machine Parameters")
    machine_type = st.sidebar.selectbox("Machine Type", ["rotating", "reciprocating"])
    machine_class = st.sidebar.selectbox("Machine Class", ["class_1 (Small)", "class_2 (Medium)", "class_3 (Large)", "class_4 (Turbo)"])
    fs = st.sidebar.number_input("Sampling Rate (Hz)", value=12000)

    # Otomatik algılanan RPM değeri varsa onu varsayılan yap
    detected_rpm_val = st.session_state.get("detected_rpm")
    default_rpm_int = int(round(float(detected_rpm_val))) if detected_rpm_val else 1797
    rpm = st.sidebar.number_input("RPM (Devir Hızı)", value=default_rpm_int, help="Dosya başlığından otomatik algılanır veya elle ayarlanabilir.")
    if detected_rpm_val:
        st.sidebar.caption(f"⚡ Dosyadan Okunan RPM: **{detected_rpm_val:.1f}**")

    # Drive End (DE) Rulman Geometrisi (Varsayılan: SKF 6205)
    de_n_balls = 9
    de_ball_diameter = 7.94
    de_pitch_diameter = 39.04
    de_contact_angle_deg = 0.0

    # Fan End (FE) Rulman Geometrisi (Varsayılan: SKF 6203)
    fe_n_balls = 9
    fe_ball_diameter = 6.75
    fe_pitch_diameter = 28.50
    fe_contact_angle_deg = 0.0

    if machine_type == "rotating":
        with st.sidebar.expander("Drive End (DE) Rulmanı [SKF 6205]", expanded=False):
            de_n_balls = st.number_input("DE Bilya Sayısı (N)", value=9, key="de_n_balls")
            de_ball_diameter = st.number_input("DE Bilya Çapı (mm)", value=7.94, key="de_bd")
            de_pitch_diameter = st.number_input("DE Hatve Çapı (mm)", value=39.04, key="de_pd")
            de_contact_angle_deg = st.number_input("DE Temas Açısı (°)", value=0.0, key="de_ca")

        with st.sidebar.expander("Fan End (FE) Rulmanı [SKF 6203]", expanded=False):
            fe_n_balls = st.number_input("FE Bilya Sayısı (N)", value=9, key="fe_n_balls")
            fe_ball_diameter = st.number_input("FE Bilya Çapı (mm)", value=6.75, key="fe_bd")
            fe_pitch_diameter = st.number_input("FE Hatve Çapı (mm)", value=28.50, key="fe_pd")
            fe_contact_angle_deg = st.number_input("FE Temas Açısı (°)", value=0.0, key="fe_ca")

    n_cylinders = 4
    stroke_type = "4-stroke"

    if machine_type == "reciprocating":
        with st.sidebar.expander("Reciprocating Parameters"):
            n_cylinders = st.number_input("Number of Cylinders", value=4)
            stroke_type = st.selectbox("Stroke Type", ["4-stroke", "2-stroke"])

    st.sidebar.header("Signal Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Signal", type=["csv", "wav", "mat"])
    if uploaded_file:
        st.sidebar.info(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

    st.sidebar.header("Quick Actions")
    quick_analysis_requested = st.sidebar.button("🔍 Quick DSP Analysis")

    return {
        "session_id": st.session_state["session_id"],
        "machine_type": machine_type,
        "machine_class": machine_class,
        "fs": fs,
        "rpm": rpm,
        "n_balls": de_n_balls,
        "ball_diameter": de_ball_diameter,
        "pitch_diameter": de_pitch_diameter,
        "contact_angle_deg": de_contact_angle_deg,
        "fe_bearing": {
            "n_balls": fe_n_balls,
            "ball_diameter": fe_ball_diameter,
            "pitch_diameter": fe_pitch_diameter,
            "contact_angle_deg": fe_contact_angle_deg,
        },
        "n_cylinders": n_cylinders,
        "stroke_type": stroke_type,
        "uploaded_file": uploaded_file,
        "quick_analysis_requested": quick_analysis_requested,
    }
