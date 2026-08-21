"""
VibraDiag API — LangGraph Orchestration & SSE Streaming Service.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException
from loguru import logger

from main_graph import build_main_graph
from src.api.schemas.common import PlotFormat, ResponseStatus
from src.api.schemas.dsp_models import ISOSeverityDTO, TimeDomainStatsDTO
from src.api.schemas.requests import DiagnosisRequest
from src.api.schemas.responses import (
    CheckerResultDTO,
    DiagnosisResponse,
    PassageDTO,
    VerificationMetricsDTO,
    VisualEvidenceDTO,
)
from src.api.services.session_service import SessionService
from src.api.services.signal_service import SignalService


class GraphService:
    """Service wrapping compiled LangGraph execution, state normalization, and SSE streaming."""

    def __init__(self, signal_service: SignalService, session_service: SessionService):
        self.signal_service = signal_service
        self.session_service = session_service
        self.checkpointer = session_service.get_checkpointer()
        self.graph = build_main_graph(checkpointer=self.checkpointer)

    def _prepare_initial_state(
        self,
        request: DiagnosisRequest,
        session_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prepares and validates initial state and config for graph execution."""
        signal_file_path = None
        loaded_signal = None
        sanitized_meta: dict[str, Any] = {}

        if request.signal_id:
            resolved_path = self.signal_service.get_signal_path(request.signal_id)
            signal_file_path = str(resolved_path)
            _, sanitized_meta = self.signal_service.validate_and_load_signal(
                resolved_path, request.machine_metadata
            )
        elif request.signal_file_path:
            resolved_path = self.signal_service.get_signal_path(request.signal_file_path)
            signal_file_path = str(resolved_path)
            _, sanitized_meta = self.signal_service.validate_and_load_signal(
                resolved_path, request.machine_metadata
            )
        elif request.machine_metadata:
            sanitized_meta = request.machine_metadata.model_dump(exclude_none=True)

        user_query = (request.query or "").strip()

        initial_state: dict[str, Any] = {
            "session_id": session_id,
            "user_query": user_query,
            "signal_file_path": signal_file_path,
            "machine_metadata": sanitized_meta,
            "errors": [],
            "messages": [],
        }

        config = {"configurable": {"thread_id": session_id}}
        return initial_state, config

    def _format_diagnosis_response(
        self,
        final_state: dict[str, Any],
        plot_format: PlotFormat,
        session_id: str,
    ) -> DiagnosisResponse:
        """Converts raw VibraDiagMainState dictionary to typed DiagnosisResponse DTO."""
        route_decision = final_state.get("route_decision", "unknown")
        llm_response = final_state.get("llm_response", "").strip()
        errors = final_state.get("errors") or []

        primary_fault = final_state.get("primary_fault")
        primary_fault_data = final_state.get("primary_fault_data")

        td_raw = final_state.get("time_domain_stats") or {}
        time_domain_dto = None
        if td_raw:
            rms_val = td_raw.get("rms") or td_raw.get("overall_rms") or td_raw.get("acceleration_rms") or 0.0
            peak_val = td_raw.get("peak") or td_raw.get("peak_value") or 0.0
            time_domain_dto = TimeDomainStatsDTO(
                rms=float(rms_val),
                peak=float(peak_val),
                crest_factor=float(td_raw.get("crest_factor", 0.0)),
                kurtosis=float(td_raw.get("kurtosis", 0.0)),
                skewness=td_raw.get("skewness"),
                margin_factor=td_raw.get("margin_factor"),
                shape_factor=td_raw.get("shape_factor"),
            )

        iso_dto = None
        severity_alert = None
        sig_res = final_state.get("signal_processing_result") or {}
        iso_zone = sig_res.get("iso_severity_zone") or sig_res.get("iso_zone")
        if iso_zone:
            iso_dto = ISOSeverityDTO(
                zone=str(iso_zone),
                meaning=str(sig_res.get("iso_severity_meaning", "")),
                rms_mm_s=float(sig_res.get("vibration_velocity_rms_mm_s", 0.0)),
                machine_class=str(sig_res.get("machine_class", "")),
                machine_description=str(sig_res.get("iso_severity_machine_description", "")),
                range_mm_s=list(sig_res.get("iso_severity_range_mm_s", [])),
                severity_table=sig_res.get("severity_table"),
            )
            if str(iso_zone).upper() == "D":
                severity_alert = "EMERGENCY_STOP_RECOMMENDED (ISO Zone D: Unacceptable Vibration Severity)"

        plots_raw = final_state.get("diagnostic_plots") or {}
        formatted_plots = self.signal_service._format_plots(plots_raw, plot_format)

        raw_passages = final_state.get("text_passages") or []
        retrieved_passages = []
        for p in raw_passages:
            if isinstance(p, dict):
                retrieved_passages.append(
                    PassageDTO(
                        id=str(p.get("id", "N/A")),
                        text=str(p.get("text", "")),
                        score=float(p.get("score", 0.0)),
                        metadata=p.get("metadata") or {},
                    )
                )

        raw_visual = final_state.get("visual_evidence") or []
        visual_evidence = []
        for v in raw_visual:
            if isinstance(v, dict):
                visual_evidence.append(
                    VisualEvidenceDTO(
                        id=str(v.get("id", "N/A")),
                        text=str(v.get("text", "")),
                        score=float(v.get("score", 0.0)),
                        metadata=v.get("metadata") or {},
                    )
                )

        verification_dto = None
        checker_results_raw = final_state.get("checker_results") or {}
        if checker_results_raw or final_state.get("is_flagged") is not None:
            checkers_mapped = {}
            for name, chk in checker_results_raw.items():
                if isinstance(chk, dict):
                    checkers_mapped[name] = CheckerResultDTO(
                        checker_name=chk.get("checker_name", name),
                        passed=bool(chk.get("passed", False)),
                        score=float(chk.get("score", 0.0)),
                        rationale=str(chk.get("rationale", "")),
                        latency_ms=float(chk.get("latency_ms", 0.0)),
                    )
            verification_dto = VerificationMetricsDTO(
                is_flagged=bool(final_state.get("is_flagged", False)),
                flag_reason=final_state.get("flag_reason"),
                checker_results=checkers_mapped,
                corrector_metrics=final_state.get("corrector_metrics") or {},
            )

        status = ResponseStatus.ERROR if route_decision == "error" else (
            ResponseStatus.WARNING if final_state.get("is_flagged") else ResponseStatus.SUCCESS
        )

        return DiagnosisResponse(
            status=status,
            session_id=session_id,
            route_decision=route_decision,
            answer=llm_response,
            primary_fault=primary_fault,
            primary_fault_data=primary_fault_data,
            iso_severity=iso_dto,
            severity_alert=severity_alert,
            time_domain_stats=time_domain_dto,
            diagnostic_plots=formatted_plots,
            retrieved_passages=retrieved_passages,
            visual_evidence=visual_evidence,
            sub_queries=final_state.get("sub_queries"),
            verification=verification_dto,
            errors=errors,
        )

    async def execute_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResponse:
        """Executes full VibraDiag pipeline asynchronously (in thread pool to avoid blocking)."""
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        initial_state, config = self._prepare_initial_state(request, session_id)

        try:
            # Run CPU-bound & I/O heavy LangGraph execution in threadpool
            final_state = await asyncio.to_thread(self.graph.invoke, initial_state, config)
            return self._format_diagnosis_response(final_state, request.plot_format, session_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Graph execution failed for session '{session_id}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Teşhis işlemi sırasında sunucu hatası: {e}") from e

    async def stream_diagnosis_events(
        self, request: DiagnosisRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Streams execution steps and node results as Server-Sent Events (SSE).
        """
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        try:
            initial_state, config = self._prepare_initial_state(request, session_id)
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error_message": str(e), "session_id": session_id}),
            }
            return

        accumulated_state: dict[str, Any] = dict(initial_state)

        yield {
            "event": "stage_start",
            "data": json.dumps({"stage": "input_router", "session_id": session_id}),
        }

        try:
            for chunk in self.graph.stream(initial_state, config, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    accumulated_state.update(node_update)

                    yield {
                        "event": "stage_update",
                        "data": json.dumps({
                            "stage": node_name,
                            "session_id": session_id,
                            "status": "completed",
                        }),
                    }

                    if node_name == "signal_dsp":
                        sig_res = node_update.get("signal_processing_result") or {}
                        primary_fault = node_update.get("primary_fault")
                        iso_zone = sig_res.get("iso_severity_zone")
                        yield {
                            "event": "dsp_complete",
                            "data": json.dumps({
                                "primary_fault": primary_fault,
                                "iso_severity_zone": iso_zone,
                                "rms_mm_s": sig_res.get("vibration_velocity_rms_mm_s"),
                            }),
                        }

                    elif node_name == "retrieval":
                        passages = node_update.get("text_passages") or []
                        visual = node_update.get("visual_evidence") or []
                        yield {
                            "event": "retrieval_complete",
                            "data": json.dumps({
                                "passages_count": len(passages),
                                "visual_count": len(visual),
                                "max_score": node_update.get("max_text_score", 0.0),
                            }),
                        }

                    elif node_name == "generation":
                        response_text = node_update.get("llm_response", "")
                        yield {
                            "event": "generation_complete",
                            "data": json.dumps({
                                "length": len(response_text),
                                "preview": response_text[:120],
                            }),
                        }

                    elif node_name == "self_corrector":
                        decision = node_update.get("route_decision")
                        is_flagged = node_update.get("is_flagged", False)
                        yield {
                            "event": "corrector_decision",
                            "data": json.dumps({
                                "decision": decision,
                                "is_flagged": is_flagged,
                                "attempts": node_update.get("correction_attempts"),
                            }),
                        }

                await asyncio.sleep(0.01)

            final_response = self._format_diagnosis_response(
                accumulated_state, request.plot_format, session_id
            )

            yield {
                "event": "complete",
                "data": json.dumps(final_response.model_dump()),
            }

        except Exception as e:
            logger.error(f"Streaming failed for session '{session_id}': {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error_message": str(e), "session_id": session_id}),
            }
