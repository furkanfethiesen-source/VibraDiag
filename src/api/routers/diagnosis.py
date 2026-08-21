"""
VibraDiag API — Diagnostic Orchestration & SSE Streaming Router.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import get_graph_service, get_signal_service
from src.api.schemas.common import PlotFormat
from src.api.schemas.requests import DiagnosisRequest, MachineMetadataDTO
from src.api.schemas.responses import DiagnosisResponse
from src.api.services.graph_service import GraphService
from src.api.services.signal_service import SignalService

router = APIRouter(prefix="/api/v1/diagnose", tags=["Fault Diagnosis & Orchestration"])


@router.post(
    "",
    response_model=DiagnosisResponse,
    summary="Execute Fault Diagnosis (JSON)",
    description=(
        "Runs the complete VibraDiag LangGraph pipeline (Signal Ingestion, DSP SubGraph, "
        "Diagnostic Plots, Hybrid RAG Retrieval, LLM Generation with ISO Zone D directive, "
        "and Self-Corrector validation loop). Accepts JSON request referencing a pre-uploaded signal_id or query."
    ),
)
async def diagnose_json(
    request: DiagnosisRequest,
    graph_svc: Annotated[GraphService, Depends(get_graph_service)],
) -> DiagnosisResponse:
    return await graph_svc.execute_diagnosis(request)


@router.post(
    "/direct",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Fault Diagnosis with Direct File Upload (Multipart)",
    description=(
        "Single-step diagnosis: Uploads a vibration signal file (.mat, .wav, .csv) and "
        "executes the complete diagnostic pipeline immediately."
    ),
)
async def diagnose_multipart(
    graph_svc: Annotated[GraphService, Depends(get_graph_service)],
    signal_svc: Annotated[SignalService, Depends(get_signal_service)],
    file: Annotated[UploadFile | None, File(description="Vibration signal file (.mat, .wav, .csv)")] = None,
    query: Annotated[str | None, Form(description="Diagnostic question (Optional if signal file uploaded)")] = None,
    session_id: Annotated[str | None, Form(description="Session thread ID (Optional)")] = None,
    fs: Annotated[float | None, Form(description="Sampling frequency in Hz (Mandatory for .mat files)")] = None,
    rpm: Annotated[float | None, Form(description="Shaft speed in RPM (Optional)")] = None,
    machine_type: Annotated[str | None, Form(description="Machine kinematics ('rotating' or 'reciprocating')")] = "rotating",
    machine_class: Annotated[str | None, Form(description="ISO Machine class ('class_1', 'class_2', 'class_3', 'class_4')")] = "class_2",
    machine_metadata_json: Annotated[str | None, Form(description="Optional JSON string containing full MachineMetadataDTO (bearing geometry, etc.)")] = None,
    plot_format: Annotated[PlotFormat, Form(description="Format of diagnostic plots ('json' or 'html')")] = PlotFormat.JSON,
) -> DiagnosisResponse:
    if not file and not query:
        raise HTTPException(
            status_code=400,
            detail="Geçerli bir kullanıcı sorusu (query) veya analiz edilecek bir sinyal dosyası (file) gereklidir.",
        )

    # 1. Save uploaded file if present
    signal_id = None
    if file and file.filename:
        upload_res = await signal_svc.save_uploaded_file(file=file, fs=fs, rpm=rpm)
        signal_id = upload_res.signal_id

    # 2. Assemble MachineMetadataDTO
    meta_dict = {}
    if machine_metadata_json:
        try:
            meta_dict = json.loads(machine_metadata_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"'machine_metadata_json' geçersiz JSON formatında: {e}") from e

    if fs is not None and "fs" not in meta_dict:
        meta_dict["fs"] = fs
    if rpm is not None and "rpm" not in meta_dict:
        meta_dict["rpm"] = rpm
    if machine_type and "machine_type" not in meta_dict:
        meta_dict["machine_type"] = machine_type
    if machine_class and "machine_class" not in meta_dict:
        meta_dict["machine_class"] = machine_class

    meta_dto = MachineMetadataDTO(**meta_dict) if meta_dict else None

    req = DiagnosisRequest(
        query=query,
        signal_id=signal_id,
        session_id=session_id,
        machine_metadata=meta_dto,
        plot_format=plot_format,
    )

    return await graph_svc.execute_diagnosis(req)


@router.post(
    "/stream",
    summary="Stream Fault Diagnosis Events (SSE)",
    description=(
        "Executes diagnosis and streams milestone events (stage_start, dsp_complete, "
        "retrieval_complete, generation_complete, corrector_decision, and complete) "
        "over Server-Sent Events (SSE, text/event-stream)."
    ),
)
async def stream_diagnosis(
    request: DiagnosisRequest,
    graph_svc: Annotated[GraphService, Depends(get_graph_service)],
) -> EventSourceResponse:
    event_generator = graph_svc.stream_diagnosis_events(request)
    return EventSourceResponse(event_generator)
