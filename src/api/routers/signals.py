"""
VibraDiag API — Deterministic Signal Processing Router.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from src.api.dependencies import get_signal_service
from src.api.schemas.dsp_models import DiagnosticPlotDTO
from src.api.schemas.requests import SignalAnalyzeRequest
from src.api.schemas.responses import SignalAnalysisResponse, SignalUploadResponse
from src.api.services.signal_service import SignalService

router = APIRouter(prefix="/api/v1/signals", tags=["Signal Processing & DSP"])


@router.post(
    "/upload",
    response_model=SignalUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Signal File",
    description="Uploads a vibration signal (.mat, .wav, .csv), validates file headers, and stores on disk.",
)
async def upload_signal_file(
    signal_svc: Annotated[SignalService, Depends(get_signal_service)],
    file: Annotated[UploadFile, File(description="Vibration signal file (.mat, .wav, .csv)")],
    fs: Annotated[float | None, Form(description="Sampling frequency in Hz (Mandatory for CWRU .mat files)")] = None,
    rpm: Annotated[float | None, Form(description="Shaft speed in RPM (Optional)")] = None,
) -> SignalUploadResponse:
    return await signal_svc.save_uploaded_file(file=file, fs=fs, rpm=rpm)


@router.post(
    "/analyze",
    response_model=SignalAnalysisResponse,
    summary="Run Deterministic DSP Analysis",
    description="Executes isolated SignalProcessingSubGraph without LLM generation (0 token cost). Returns time domain stats, envelope harmonics, and ISO severity.",
)
async def analyze_signal(
    request: SignalAnalyzeRequest,
    signal_svc: Annotated[SignalService, Depends(get_signal_service)],
) -> SignalAnalysisResponse:
    target_ref = request.signal_id or request.signal_file_path
    if not target_ref:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="'signal_id' veya 'signal_file_path' belirtilmelidir.")

    return await asyncio.to_thread(
        signal_svc.run_isolated_dsp,
        signal_file_path=target_ref,
        meta_dto=request.machine_metadata,
        plot_format=request.plot_format,
    )


@router.post(
    "/plots",
    response_model=DiagnosticPlotDTO,
    summary="Generate Diagnostic Plots",
    description="Generates FFT Spectrum and Spectral Kurtogram Heatmap figures as Plotly JSON or standalone HTML.",
)
async def generate_signal_plots(
    request: SignalAnalyzeRequest,
    signal_svc: Annotated[SignalService, Depends(get_signal_service)],
) -> DiagnosticPlotDTO:
    target_ref = request.signal_id or request.signal_file_path
    if not target_ref:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="'signal_id' veya 'signal_file_path' belirtilmelidir.")

    analysis = await asyncio.to_thread(
        signal_svc.run_isolated_dsp,
        signal_file_path=target_ref,
        meta_dto=request.machine_metadata,
        plot_format=request.plot_format,
    )
    return analysis.diagnostic_plots or DiagnosticPlotDTO()
