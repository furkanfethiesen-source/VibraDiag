"""
Centralized LangGraph State Definitions for VibraDiag.

Houses top-level application state (VibraDiagMainState) and domain-specific
subgraph internal states (SignalProcessingInternalState).
"""

from __future__ import annotations

import operator
from typing import Any, Literal, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from deterministic_tools.reader import LoadedSignal


class VibraDiagMainState(TypedDict, total=False):
    """
    Global shared state for VibraDiag LangGraph pipeline.
    """

    session_id: str
    user_query: str
    signal_file_path: str | None

    messages: Annotated[list[BaseMessage], add_messages]

    machine_metadata: dict[str, Any]

    needs_signal_context: bool
    route_decision: Literal["signal_only", "rag_only", "hybrid", "general_chat", "error"]

    signal_analysis: dict[str, Any] | None
    signal_processing_result: dict[str, Any]
    diagnostic_plots: dict[str, dict]

    text_passages: list[dict[str, Any]]
    visual_evidence: list[dict[str, Any]]
    merged_context: str

    llm_response: str

    errors: Annotated[list[str], operator.add]

    correction_attempts: dict[str, int]
    excluded_chunk_ids: Annotated[list[str], operator.add]
    reformulated_query: str | None
    previous_failure_reason: str | None
    is_flagged: bool
    flag_reason: str | None
    corrector_metrics: dict[str, Any]
    corrector_history: list[dict[str, Any]]


class SignalProcessingInternalState(TypedDict, total=False):
    """
    Local internal state for SignalProcessingSubGraph.
    """

    machine_type: Literal["rotating", "reciprocating"]
    loaded_signal: LoadedSignal
    rpm: float

    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle_deg: float

    primary_channel: str
    kurtogram_band: tuple
    kurtogram: list
    kurtogram_reliable: bool
    final_band: tuple
    band_source: str
    band_filter_used: int | None
    envelope_diagnosis: dict
    fault_localization: dict
    direct_spectrum_diagnosis: dict

    n_cylinders: int
    stroke_type: str

    reciprocating_diagnosis: dict

    machine_class: str
    vibration_velocity_rms_mm_s: float
    severity_table: dict
    iso_severity_zone: str
    iso_severity_meaning: str
    iso_severity_range_mm_s: tuple
    iso_severity_machine_description: str


class RetrievalState(TypedDict, total=False):
    """
    LangGraph state schema for retrieval orchestration (RetrievalGraph).
    """

    query: str
    text_top_k: int
    visual_top_k: int
    threshold: float
    deduplicate_parents: bool
    text_results: dict[str, Any]
    visual_results: dict[str, Any]
    max_text_score: float
    visual_triggered: bool
    text_passages: list[dict[str, Any]]
    visual_evidence: list[dict[str, Any]]

    is_complex: bool
    classifier_confidence: float
    sub_queries: list[str]
    decomposer_reasoning: str | None
    sub_query_passages: dict[str, list[dict[str, Any]]]

    excluded_chunk_ids: Annotated[list[str], operator.add]
    reformulated_query: str | None
    attempt_number: int




