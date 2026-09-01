"""
VibraDiag API — Response DTO Models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.api.schemas.common import ResponseStatus
from src.api.schemas.dsp_models import DiagnosticPlotDTO, ISOSeverityDTO, TimeDomainStatsDTO


class PassageDTO(BaseModel):
    id: str = Field(..., description="Chunk or Parent Document ID")
    text: str = Field(..., description="Retrieved passage textual content")
    score: float = Field(default=0.0, description="Reranker relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata tags (source, page, section)")


class VisualEvidenceDTO(BaseModel):
    id: str = Field(..., description="Visual chunk ID")
    text: str = Field(default="", description="Visual reasoning and structural description")
    score: float = Field(default=0.0, description="Vector similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Visual metadata (content_type, page, fault_type)")


class CheckerResultDTO(BaseModel):
    checker_name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    latency_ms: float = 0.0


class VerificationMetricsDTO(BaseModel):
    is_flagged: bool = False
    flag_reason: str | None = None
    checker_results: dict[str, CheckerResultDTO] = Field(default_factory=dict)
    corrector_metrics: dict[str, Any] = Field(default_factory=dict)


class DiagnosisResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    session_id: str = Field(..., description="Active session ID")
    route_decision: str = Field(..., description="Route executed ('hybrid', 'signal_only', 'rag_only', etc.)")
    answer: str = Field(..., description="Synthesized diagnostic response, root-cause explanation, and maintenance recommendations")

    primary_fault: str | None = Field(default=None, description="Identified primary mechanical or bearing fault")
    primary_fault_data: dict[str, Any] | None = Field(default=None, description="Detailed fault confidence, harmonics, sidebands")
    is_compound_fault: bool = Field(default=False, description="True if multiple interacting faults (e.g. BSF + BPFI) are co-occurring")
    secondary_fault_name: str | None = Field(default=None, description="Co-occurring / secondary fault name if compound defect")
    secondary_fault_abbr: str | None = Field(default=None, description="Secondary fault abbreviation (e.g. BPFI, BSF)")
    co_occurring_faults: list[dict[str, Any]] = Field(default_factory=list, description="List of co-occurring strong secondary fault candidates")
    detected_rpm: float | None = Field(default=None, description="Automatically extracted/evaluated shaft speed in RPM")
    iso_severity: ISOSeverityDTO | None = Field(default=None, description="ISO 2372 / 10816 evaluation")
    severity_alert: str | None = Field(default=None, description="Emergency directive if in Danger zone (Zone D)")

    time_domain_stats: TimeDomainStatsDTO | None = Field(default=None, description="Vibration metrics (RMS, Kurtosis, Crest Factor, Peak)")
    diagnostic_plots: DiagnosticPlotDTO | None = Field(default=None, description="Serialized FFT Spectrum and Kurtogram figures")

    retrieved_passages: list[PassageDTO] = Field(default_factory=list, description="Retrieved RAG knowledge passages")
    visual_evidence: list[VisualEvidenceDTO] = Field(default_factory=list, description="Visual diagnostic evidence (spectra, tables)")
    sub_queries: list[str] | None = Field(default=None, description="Sub-queries produced by query decomposer")
    verification: VerificationMetricsDTO | None = Field(default=None, description="Self-corrector verification outcome")

    errors: list[str] = Field(default_factory=list, description="List of non-fatal or diagnostic warnings/errors")


class SignalUploadResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    signal_id: str = Field(..., description="Unique ID for referencing this uploaded signal")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Stored file path on server")
    size_bytes: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Detected format (.wav, .mat, .csv)")
    channels: list[str] = Field(default_factory=list, description="Detected channels in file (e.g. ['DE', 'FE', 'BA'])")
    fs: float | None = Field(default=None, description="Detected or provided sampling rate in Hz")
    rpm: float | None = Field(default=None, description="Detected or extracted RPM")
    duration_seconds: float | None = Field(default=None, description="Signal duration in seconds")
    message: str = Field(default="Signal uploaded successfully")


class SignalAnalysisResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    signal_id: str | None = None
    primary_fault: str | None = None
    primary_fault_data: dict[str, Any] | None = None
    is_compound_fault: bool = False
    secondary_fault_name: str | None = None
    secondary_fault_abbr: str | None = None
    co_occurring_faults: list[dict[str, Any]] = Field(default_factory=list)
    detected_rpm: float | None = None
    iso_severity: ISOSeverityDTO | None = None
    severity_alert: str | None = None
    time_domain_stats: TimeDomainStatsDTO | None = None
    dsp_analysis: dict[str, Any] = Field(default_factory=dict, description="Raw deterministic DSP details")
    diagnostic_plots: DiagnosticPlotDTO | None = None


class RetrievalResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    query: str
    is_complex: bool = False
    sub_queries: list[str] = Field(default_factory=list)
    max_text_score: float = 0.0
    visual_triggered: bool = False
    text_passages: list[PassageDTO] = Field(default_factory=list)
    visual_evidence: list[VisualEvidenceDTO] = Field(default_factory=list)
    merged_context: str = ""


class DecomposeResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    query: str
    is_complex: bool
    classifier_confidence: float
    sub_queries: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class SessionStateResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    session_id: str
    message_count: int
    route_decision: str | None = None
    primary_fault: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str = "VibraDiag"
    version: str = "0.1.0"
    timestamp: str
    components: dict[str, Any] = Field(default_factory=dict)


class StreamEventDTO(BaseModel):
    event: str = Field(..., description="SSE event name (e.g. stage_start, dsp_done, token, complete)")
    data: dict[str, Any] = Field(default_factory=dict, description="Payload data for the event")


class RecentSessionSummaryDTO(BaseModel):
    session_id: str
    message_count: int = 0
    primary_fault: str | None = None
    last_query_preview: str | None = None
    last_checkpoint_id: str | None = None


class RecentSessionsListResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.SUCCESS
    count: int
    sessions: list[RecentSessionSummaryDTO] = Field(default_factory=list)
