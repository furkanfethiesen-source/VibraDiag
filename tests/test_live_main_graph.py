"""
VibraDiag — Live End-to-End Main Graph Test Script.
===================================================

Executes the full compiled StateGraph with live Qdrant, DocStore, Groq LLM,
and Gemini Judge evaluators.

Supports:
1. 'signal_only' (Default): Runs CWRU bearing dataset (e.g. data/signal_test/105.mat),
generates interactive Plotly HTML diagnostic plots, and saves structured
diagnosis report to sample_outputs/signal_only.json.
2. 'hybrid': Runs signal analysis combined with a specific user query, saving
retrieved chunks and diagnosis to sample_outputs/sample_hybrid.json.
3. 'rag_only': Runs pure text documentation query, saving answer and retrieved
chunks to sample_outputs/sample_rag_only.json.
Usage:
------
uv run python tests/test_live_main_graph.py --mode signal_only
uv run python tests/test_live_main_graph.py --mode hybrid
uv run python tests/test_live_main_graph.py --mode rag_only
uv run python tests/test_live_main_graph.py --mode all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import plotly.graph_objects as go
from loguru import logger

from src.config_loader import load_appcfg
from src.main_graph import build_main_graph
from src.schemas.states import VibraDiagMainState

_app_cfg = load_appcfg()
_paths_cfg = getattr(_app_cfg, "paths", {}) or {}

SAMPLE_OUTPUTS_DIR = PROJECT_ROOT / _paths_cfg.get("sample_outputs_dir", "sample_outputs")
SAMPLE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SIGNAL_PATH = _paths_cfg.get("test_signal_path", "data/signal_test/105.mat")
DEFAULT_SIGNAL_ONLY_JSON = PROJECT_ROOT / _paths_cfg.get("sample_signal_only_json", "sample_outputs/signal_only.json")
DEFAULT_HYBRID_JSON = PROJECT_ROOT / _paths_cfg.get("sample_hybrid_json", "sample_outputs/sample_hybrid.json")
DEFAULT_RAG_ONLY_JSON = PROJECT_ROOT / _paths_cfg.get("sample_rag_only_json", "sample_outputs/sample_rag_only.json")


def save_interactive_plots(
    plots_dict: dict[str, Any] | None,
    output_dir: Path,
    prefix: str = "signal_only",
) -> dict[str, str]:
    """
    Extracts existing Plotly interactive HTML file paths, saving figures only if not already saved.
    """
    if not plots_dict:
        logger.warning("No diagnostic plots found in state to save.")
        return {}

    saved_files = {}

    has_existing_paths = False
    for plot_name, plot_val in plots_dict.items():
        if isinstance(plot_val, (str, Path)):
            p = Path(plot_val)
            if p.exists():
                has_existing_paths = True
                try:
                    rel_path = str(p.relative_to(PROJECT_ROOT))
                except ValueError:
                    rel_path = str(p)
                saved_files[f"{plot_name}_html" if not plot_name.endswith("_html") and not plot_name.endswith("_path") else plot_name] = rel_path

    if not has_existing_paths:
        for plot_name, plot_val in plots_dict.items():
            if isinstance(plot_val, dict):
                try:
                    fig = go.Figure(plot_val)
                    html_filename = f"{prefix}_{plot_name}.html"
                    html_path = output_dir / html_filename
                    fig.write_html(str(html_path), include_plotlyjs="cdn")
                    logger.info("Saved interactive plot: %s", html_path)
                    saved_files[f"{plot_name}_html"] = str(html_path.relative_to(PROJECT_ROOT))
                except Exception as e:
                    logger.error("Failed to save plot '%s': %s", plot_name, e)

    return saved_files


from src.deterministic_tools.fault_analyzer import pick_primary_fault


def extract_detected_metadata(
    state: dict,
    source_path: str | None = None,
    saved_plots: dict[str, str] | None = None,
) -> dict:
    """
    Extracts diagnostic findings and telemetry metrics from state.
    """
    signal_res = state.get("signal_processing_result") or state.get("signal_analysis") or {}

    envelope_diagnosis = signal_res.get("envelope_diagnosis") or {}
    fault_localization = signal_res.get("fault_localization") or {}
    direct_spectrum_diagnosis = signal_res.get("direct_spectrum_diagnosis") or {}
    reciprocating_diagnosis = signal_res.get("reciprocating_diagnosis") or {}

    fault_summary = pick_primary_fault(
        direct_spectrum_diagnosis=direct_spectrum_diagnosis,
        envelope_diagnosis=envelope_diagnosis,
        reciprocating_diagnosis=reciprocating_diagnosis,
    )

    primary_fault = state.get("primary_fault") or fault_summary.get("primary_fault_name")
    primary_fault_data = state.get("primary_fault_data") or fault_summary

    time_domain_stats = (
        state.get("time_domain_stats")
        or signal_res.get("time_domain_stats")
        or {}
    )

    metadata = {
        "route_decision": state.get("route_decision"),
        "auto_generated_query": state.get("user_query"),

        "primary_fault": primary_fault,
        "primary_fault_data": primary_fault_data,

        "time_domain_stats": time_domain_stats,
        "direct_spectrum_diagnosis": direct_spectrum_diagnosis,
        "envelope_diagnosis": envelope_diagnosis,
        "fault_localization": fault_localization,
        "reciprocating_diagnosis": reciprocating_diagnosis,

        "iso_severity_zone": signal_res.get("iso_severity_zone"),
        "iso_severity_meaning": signal_res.get("iso_severity_meaning"),
        "iso_severity_range_mm_s": signal_res.get("iso_severity_range_mm_s"),

        "rpm": signal_res.get("rpm") or state.get("machine_metadata", {}).get("rpm"),
        "sampling_frequency_hz": state.get("machine_metadata", {}).get("fs", 12000.0),
        "vibration_velocity_rms_mm_s": signal_res.get("vibration_velocity_rms_mm_s"),

        "kurtogram_reliable": signal_res.get("kurtogram_reliable"),
        "band_source": signal_res.get("band_source"),
        "primary_channel": signal_res.get("primary_channel"),

        "self_corrector_decision": state.get("route_decision"),
        "checker_results": state.get("checker_results", {}),
        "correction_attempts": state.get("correction_attempts", {}),
        "is_flagged": state.get("is_flagged", False),
        "flag_reason": state.get("flag_reason"),
        "corrector_history": state.get("corrector_history", []),
        "corrector_metrics": state.get("corrector_metrics", {}),

        "saved_plots": saved_plots or {},
    }

    return metadata


def run_signal_only_test(
    signal_path: str = DEFAULT_SIGNAL_PATH,
    output_json_path: Path = DEFAULT_SIGNAL_ONLY_JSON,
) -> dict[str, Any]:
    """
    Runs full pipeline in Signal Only mode.
    """
    logger.info("=" * 70)
    logger.info("🚀 RUNNING SIGNAL ONLY TEST: %s", signal_path)
    logger.info("=" * 70)

    machine_metadata = {
        "machine_name": "CWRU Test Rig 12k DE",
        "machine_type": "rotating",
        "machine_class": "Class II",
        "rpm": 1797.0,
        "fs": 12000.0,
        "n_balls": 9,
        "ball_diameter": 7.94,
        "pitch_diameter": 39.04,
        "contact_angle_deg": 0.0,
        "vibration_velocity_rms_mm_s": 4.25, 
    }

    initial_state: VibraDiagMainState = {
        "session_id": "session_signal_only_live",
        "signal_file_path": signal_path,
        "user_query": "",
        "machine_metadata": machine_metadata,
    }

    app = build_main_graph()

    config = {"configurable": {"thread_id": "live_signal_only_thread"}}
    final_state = app.invoke(initial_state, config=config)

    # Save Plots
    diagnostic_plots = final_state.get("diagnostic_plots", {})
    saved_plots = save_interactive_plots(
        plots_dict=diagnostic_plots,
        output_dir=SAMPLE_OUTPUTS_DIR,
        prefix="signal_only",
    )

    detected_meta = extract_detected_metadata(
        state=final_state,
        source_path=signal_path,
        saved_plots=saved_plots,
    )

    output_payload = {
        "mode": "signal_only",
        "source_path": signal_path,
        "generated_answer": final_state.get("llm_response", ""),
        "detected_metadata": detected_meta,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    logger.success("Signal Only test completed! Output saved to: %s", output_json_path)

    print("\n" + "=" * 70)
    print("📊 VIBRADIAG SIGNAL ONLY TEST REPORT")
    print("=" * 70)
    print(f"📁 Sinyal Dosyası: {signal_path}")
    primary = detected_meta.get('primary_fault')
    primary_data = detected_meta.get('primary_fault_data') or {}
    print(f"⚙️  Tespit Edilen Arıza: {primary} "
        f"(severity={primary_data.get('severity')}, confidence={primary_data.get('confidence')}, "
        f"kaynak={primary_data.get('source')})")
    print(f"🏷️  ISO Şiddet Zonu: Bölge {detected_meta.get('iso_severity_zone')} ({detected_meta.get('iso_severity_meaning')})")
    print(f"🛡️  Self-Corrector Kararı: {detected_meta.get('self_corrector_decision')} (Flagged: {detected_meta.get('is_flagged')})")
    print(f"📈 Kaydedilen Grafikler: {list(saved_plots.values())}")
    print("-" * 70)
    print(f"📝 Otomatik Oluşturulan Soru:\n{detected_meta.get('auto_generated_query')}")
    print("-" * 70)
    print(f"🤖 LLM Teşhis Yanıtı:\n{output_payload['generated_answer'][:600]}...\n[Tamamı {output_json_path} içinde]")
    print("=" * 70 + "\n")

    return output_payload


def run_hybrid_test(
    signal_path: str = DEFAULT_SIGNAL_PATH,
    user_query: str = "Rulmandaki arızanın boyutu nedir ve acil bakım olarak hangi adımlar atılmalıdır?",
    output_json_path: Path = DEFAULT_HYBRID_JSON,
) -> dict[str, Any]:
    """
    Runs full pipeline in Hybrid mode (Signal + User Query).
    """
    logger.info("=" * 70)
    logger.info("🚀 RUNNING HYBRID TEST: %s | Query: '%s'", signal_path, user_query)
    logger.info("=" * 70)

    machine_metadata = {
        "machine_name": "CWRU Test Rig 12k DE",
        "machine_type": "rotating",
        "machine_class": "Class II",
        "rpm": 1797.0,
        "fs": 12000.0,
        "n_balls": 9,
        "ball_diameter": 7.94,
        "pitch_diameter": 39.04,
        "contact_angle_deg": 0.0,
        "vibration_velocity_rms_mm_s": 4.25,
    }

    initial_state: VibraDiagMainState = {
        "session_id": "session_hybrid_live",
        "signal_file_path": signal_path,
        "user_query": user_query,
        "machine_metadata": machine_metadata,
    }

    app = build_main_graph()

    config = {"configurable": {"thread_id": "live_hybrid_thread"}}
    final_state = app.invoke(initial_state, config=config)

    diagnostic_plots = final_state.get("diagnostic_plots", {})
    saved_plots = save_interactive_plots(
        plots_dict=diagnostic_plots,
        output_dir=SAMPLE_OUTPUTS_DIR,
        prefix="hybrid",
    )

    detected_meta = extract_detected_metadata(
        state=final_state,
        source_path=signal_path,
        saved_plots=saved_plots,
    )

    output_payload = {
        "mode": "hybrid",
        "source_path": signal_path,
        "user_query": user_query,
        "generated_answer": final_state.get("llm_response", ""),
        "detected_metadata": detected_meta,
        "retrieved_chunks": {
            "text_passages": final_state.get("text_passages", []),
            "visual_evidence": final_state.get("visual_evidence", []),
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    logger.success("Hybrid test completed! Output saved to: %s", output_json_path)
    return output_payload


def run_rag_only_test(
    user_query: str = "ISO 10816 standartlarına göre Class II makinelerde Zone C titreşim şiddeti ne anlama gelir?",
    output_json_path: Path = DEFAULT_RAG_ONLY_JSON,
) -> dict[str, Any]:
    """
    Runs full pipeline in Pure RAG mode (Text Query only).
    """
    logger.info("=" * 70)
    logger.info("🚀 RUNNING RAG ONLY TEST: Query: '%s'", user_query)
    logger.info("=" * 70)

    initial_state: VibraDiagMainState = {
        "session_id": "session_rag_only_live",
        "user_query": user_query,
        "signal_file_path": None,
    }

    app = build_main_graph()

    config = {"configurable": {"thread_id": "live_rag_only_thread"}}
    final_state = app.invoke(initial_state, config=config)

    output_payload = {
        "mode": "rag_only",
        "user_query": user_query,
        "generated_answer": final_state.get("llm_response", ""),
        "retrieved_chunks": {
            "text_passages": final_state.get("text_passages", []),
            "visual_evidence": final_state.get("visual_evidence", []),
        },
        "detected_metadata": {
            "route_decision": final_state.get("route_decision"),
            "self_corrector_decision": final_state.get("route_decision"),
            "flag_reason": final_state.get("flag_reason"),
            "is_flagged": final_state.get("is_flagged", False),
            "checker_results": final_state.get("corrector_history", {}),
            "corrector_metrics": final_state.get("corrector_metrics", {}),
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    logger.success("RAG Only test completed! Output saved to: %s", output_json_path)
    return output_payload


def main():
    parser = argparse.ArgumentParser(description="VibraDiag Main Graph Live Test Suite")
    parser.add_argument(
        "--mode",
        choices=["signal_only", "hybrid", "rag_only", "all"],
        default="signal_only",
        help="Test execution mode (default: signal_only)",
    )
    parser.add_argument(
        "--signal_path",
        default=DEFAULT_SIGNAL_PATH,
        help=f"Path to vibration signal file (default: {DEFAULT_SIGNAL_PATH})",
    )
    args = parser.parse_args()

    if args.mode == "signal_only":
        run_signal_only_test(signal_path=args.signal_path)
    elif args.mode == "hybrid":
        run_hybrid_test(signal_path=args.signal_path)
    elif args.mode == "rag_only":
        run_rag_only_test()
    elif args.mode == "all":
        run_signal_only_test(signal_path=args.signal_path)
        run_hybrid_test(signal_path=args.signal_path)
        run_rag_only_test()


if __name__ == "__main__":
    main()
