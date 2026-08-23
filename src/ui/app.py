"""
VibraDiag — Streamlit Diagnostic Dashboard.

Main entry point for the chat-based vibration fault diagnosis UI.
Communicates with the FastAPI backend via HTTP client.

Run:
    uv run streamlit run src/ui/app.py
"""

from __future__ import annotations

import pathlib
from typing import Any

import streamlit as st
import yaml
from loguru import logger

from src.ui.api_client import VibraDiagAPIError, get_client
from src.ui.components.diagnostic_card import render_diagnostic_card, render_file_chip
from src.ui.components.header import render_header
from src.ui.components.sidebar import render_sidebar


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "app.yaml"
CSS_PATH = pathlib.Path(__file__).resolve().parent / "styles" / "custom.css"


def load_streamlit_config() -> dict[str, Any]:
    """Load streamlit section from app.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("streamlit", {})


def load_custom_css() -> None:
    """Inject custom CSS stylesheet."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize all session state keys with defaults."""
    defaults = {
        "messages": [],                 # Chat history: list[dict]
        "session_id": None,             # Backend session thread ID
        "uploaded_signal_id": None,     # Currently uploaded signal ID
        "uploaded_filename": None,      # Uploaded file display name
        "uploaded_file_info": "",       # Extra upload info string
        "last_diagnosis": None,         # Last DiagnosisResponse dict
        "system_status": "OK",          # Backend health status
        "sensor_id": "—",              # Derived from uploaded file
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)


def check_backend_health(api_base_url: str) -> str:
    """Quick health check, returns 'OK' or 'ERROR'."""
    try:
        client = get_client(api_base_url)
        resp = client.get_healthz()
        return "OK" if resp.get("status") == "ok" else "ERROR"
    except Exception:
        return "ERROR"


def upload_signal_to_backend(
    api_base_url: str,
    file_content: bytes,
    filename: str,
    fs: float | None = None,
    rpm: float | None = None,
) -> dict[str, Any] | None:
    """Upload a signal file to backend and return the response."""
    try:
        client = get_client(api_base_url)
        result = client.upload_signal(file_content, filename, fs=fs, rpm=rpm)
        return result
    except VibraDiagAPIError as e:
        st.error(f"❌ Upload failed: {e.detail or e}")
        logger.error(f"Upload failed: {e}")
        return None


def run_diagnosis(
    api_base_url: str,
    query: str,
    signal_id: str | None = None,
    session_id: str | None = None,
    machine_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Execute full diagnosis pipeline via sync endpoint."""
    try:
        client = get_client(api_base_url)
        result = client.diagnose(
            query=query,
            signal_id=signal_id,
            session_id=session_id,
            machine_metadata=machine_metadata,
            plot_format="json",
        )
        return result
    except VibraDiagAPIError as e:
        st.error(f"❌ Diagnosis failed: {e.detail or e}")
        logger.error(f"Diagnosis failed: {e}")
        return None


def load_session_history(api_base_url: str, session_id: str) -> list[dict[str, Any]]:
    """Load chat history from backend for session recovery after refresh."""
    try:
        client = get_client(api_base_url)
        messages = client.get_session_messages(session_id)
        return messages if isinstance(messages, list) else []
    except Exception:
        return []


def build_machine_metadata(sidebar_params: dict) -> dict[str, Any]:
    """Build MachineMetadataDTO dict from sidebar params."""
    # Extract clean machine_class (remove description in parentheses)
    raw_class = sidebar_params.get("machine_class", "class_2 (Medium)")
    machine_class = raw_class.split(" ")[0] if " " in raw_class else raw_class

    meta: dict[str, Any] = {
        "machine_type": sidebar_params.get("machine_type", "rotating"),
        "machine_class": machine_class,
        "fs": sidebar_params.get("fs"),
        "rpm": sidebar_params.get("rpm"),
    }

    if sidebar_params.get("machine_type") == "rotating":
        meta.update({
            "n_balls": sidebar_params.get("n_balls"),
            "ball_diameter": sidebar_params.get("ball_diameter"),
            "pitch_diameter": sidebar_params.get("pitch_diameter"),
            "contact_angle_deg": sidebar_params.get("contact_angle_deg"),
        })
    elif sidebar_params.get("machine_type") == "reciprocating":
        meta.update({
            "n_cylinders": sidebar_params.get("n_cylinders"),
            "stroke_type": sidebar_params.get("stroke_type"),
        })

    # Remove None values
    return {k: v for k, v in meta.items() if v is not None}



def render_chat_history() -> None:
    """Render all stored chat messages."""
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            # Text content
            if msg.get("text"):
                st.markdown(msg["text"])

            # File chip
            if msg.get("file_chip"):
                chip = msg["file_chip"]
                render_file_chip(chip.get("filename", ""), chip.get("info", ""))

            # Diagnostic card
            if msg.get("diagnosis"):
                render_diagnostic_card(msg["diagnosis"], card_index=idx)


def add_user_message(text: str, file_chip: dict | None = None) -> None:
    """Add a user message to chat history."""
    msg: dict[str, Any] = {"role": "user", "text": text}
    if file_chip:
        msg["file_chip"] = file_chip
    st.session_state.messages.append(msg)


def add_assistant_message(
    text: str = "",
    diagnosis: dict | None = None,
    file_chip: dict | None = None,
) -> None:
    """Add an assistant message to chat history."""
    msg: dict[str, Any] = {"role": "assistant"}
    if text:
        msg["text"] = text
    if diagnosis:
        msg["diagnosis"] = diagnosis
    if file_chip:
        msg["file_chip"] = file_chip
    st.session_state.messages.append(msg)


# ---------------------------------------------------------------------------
# File Upload Handler
# ---------------------------------------------------------------------------

def handle_file_upload(
    sidebar_params: dict,
    api_base_url: str,
) -> None:
    """Process a newly uploaded file."""
    uploaded_file = sidebar_params.get("uploaded_file")

    if uploaded_file is None:
        return

    # Avoid re-uploading the same file
    if st.session_state.uploaded_filename == uploaded_file.name:
        return

    with st.spinner(f"⏳ Uploading {uploaded_file.name}..."):
        fs = sidebar_params.get("fs")
        rpm = sidebar_params.get("rpm")
        file_bytes = uploaded_file.getvalue()

        result = upload_signal_to_backend(
            api_base_url, file_bytes, uploaded_file.name, fs=fs, rpm=rpm
        )

    if result:
        signal_id = result.get("signal_id", "")
        channels = result.get("channels", [])
        detected_fs = result.get("fs")
        duration = result.get("duration_seconds")

        st.session_state.uploaded_signal_id = signal_id
        st.session_state.uploaded_filename = uploaded_file.name

        # Build info string
        info_parts = []
        if detected_fs:
            info_parts.append(f"{int(detected_fs)} Hz")
        if channels:
            info_parts.append(f"{len(channels)} ch")
        if duration:
            info_parts.append(f"{duration:.1f}s")
        info_str = ", ".join(info_parts)
        st.session_state.uploaded_file_info = info_str

        # Derive sensor ID from filename
        stem = pathlib.Path(uploaded_file.name).stem
        st.session_state.sensor_id = stem[:8].upper()

        # Add to chat
        add_assistant_message(
            text=f"✅ Signal uploaded successfully: **{uploaded_file.name}**",
            file_chip={"filename": uploaded_file.name, "info": f"({info_str})"},
        )

        st.toast(f"✅ Uploaded: {uploaded_file.name}", icon="📁")


# ---------------------------------------------------------------------------
# Diagnosis Handler
# ---------------------------------------------------------------------------

def handle_diagnosis(
    user_query: str,
    sidebar_params: dict,
    api_base_url: str,
) -> None:
    """Run full diagnosis pipeline and display results."""
    session_id = sidebar_params.get("session_id") or st.session_state.session_id
    signal_id = st.session_state.uploaded_signal_id
    machine_metadata = build_machine_metadata(sidebar_params)

    with st.spinner("🔬 Running diagnostic pipeline..."):
        result = run_diagnosis(
            api_base_url=api_base_url,
            query=user_query,
            signal_id=signal_id,
            session_id=session_id,
            machine_metadata=machine_metadata,
        )

    if result:
        st.session_state.last_diagnosis = result

        # Extract answer text
        answer = result.get("answer", "")
        primary_fault = result.get("primary_fault", "")

        # Build a summary text before the card
        summary_parts = []
        if primary_fault:
            summary_parts.append(f"**Primary Fault:** {primary_fault}")
        if result.get("iso_severity"):
            zone = result["iso_severity"].get("zone", "")
            meaning = result["iso_severity"].get("meaning", "")
            summary_parts.append(f"**ISO Zone:** {zone} ({meaning})")

        intro_text = " │ ".join(summary_parts) if summary_parts else ""

        # Check if there is actual DSP / fault / plot data
        has_fault = bool(result.get("primary_fault") or result.get("primary_fault_data"))
        raw_plots = result.get("diagnostic_plots") or {}
        has_plots = any(v for v in raw_plots.values() if v) if isinstance(raw_plots, dict) else bool(raw_plots)

        # Add assistant message with diagnostic card only if real diagnosis data exists
        add_assistant_message(
            text=answer if answer else intro_text,
            diagnosis=result if (has_fault or has_plots) else None,
        )

        # Update header state
        if result.get("iso_severity"):
            iso = result["iso_severity"]
            # Store for header rendering
            st.session_state["_severity_zone"] = iso.get("zone")
            st.session_state["_machine_class"] = iso.get("machine_class")

        st.rerun()


def handle_quick_dsp(sidebar_params: dict, api_base_url: str) -> None:
    """Run quick DSP-only analysis without LLM."""
    signal_id = st.session_state.uploaded_signal_id
    if not signal_id:
        st.sidebar.warning("⚠️ Please upload a signal file first.")
        return

    machine_metadata = build_machine_metadata(sidebar_params)

    add_user_message("🔍 Quick DSP Analysis requested")

    with st.spinner("⚡ Running deterministic signal analysis..."):
        try:
            client = get_client(api_base_url)
            result = client.analyze_signal(
                signal_id=signal_id,
                signal_file_path="",
                machine_metadata=machine_metadata,
                plot_format="json",
            )
        except VibraDiagAPIError as e:
            st.error(f"❌ Analysis failed: {e.detail or e}")
            return

    if result:
        st.session_state.last_diagnosis = result
        add_assistant_message(
            text="⚡ **Quick DSP Analysis Complete** (no LLM — zero token cost)",
            diagnosis=result,
        )
        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point."""
    # Load config
    st_config = load_streamlit_config()
    api_base_url = st_config.get("api_base_url", "http://localhost:8000")

    # Page config
    st.set_page_config(
        page_title=st_config.get("page_title", "VibraDiag"),
        page_icon=st_config.get("page_icon", "⚡"),
        layout=st_config.get("layout", "wide"),
        initial_sidebar_state="expanded",
    )

    # Initialize
    load_custom_css()
    init_session_state()

    # Check backend health (cached per session)
    if "backend_status_checked" not in st.session_state:
        st.session_state.system_status = check_backend_health(api_base_url)
        st.session_state.backend_status_checked = True

    # Sidebar
    sidebar_params = render_sidebar(api_base_url=api_base_url)

    # Sync session_id
    if sidebar_params.get("session_id"):
        st.session_state.session_id = sidebar_params["session_id"]

    # Session recovery: load history from backend on first load
    if (
        st.session_state.session_id
        and not st.session_state.messages
        and st.session_state.system_status == "OK"
    ):
        backend_messages = load_session_history(api_base_url, st.session_state.session_id)
        if backend_messages:
            for msg in backend_messages:
                role = msg.get("type", "ai")
                content = msg.get("content", "")
                if role in ("human", "HumanMessage"):
                    add_user_message(content)
                elif content:
                    add_assistant_message(text=content)

    # Header
    render_header(
        system_status=st.session_state.system_status,
        sensor_id=st.session_state.sensor_id,
        machine_class=st.session_state.get("_machine_class"),
        severity_zone=st.session_state.get("_severity_zone"),
    )

    # Handle file upload
    handle_file_upload(sidebar_params, api_base_url)

    # Handle quick DSP
    if sidebar_params.get("quick_analysis_requested"):
        handle_quick_dsp(sidebar_params, api_base_url)

    # Render chat history
    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Describe the issue or upload data..."):
        add_user_message(prompt)

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run diagnosis
        handle_diagnosis(prompt, sidebar_params, api_base_url)


if __name__ == "__main__":
    main()
