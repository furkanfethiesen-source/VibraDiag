"""
VibraDiag API — Request DTO Models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.api.schemas.common import PlotFormat


class MachineMetadataDTO(BaseModel):
    machine_type: Literal["rotating", "reciprocating"] | None = Field(
        default="rotating", description="Machine kinematics type ('rotating' or 'reciprocating')"
    )
    machine_class: Literal["class_1", "class_2", "class_3", "class_4"] | str | None = Field(
        default="class_2", description="ISO standard machine class"
    )
    rpm: float | None = Field(
        default=None, description="Shaft rotational speed in RPM"
    )
    fs: float | None = Field(
        default=None, description="Sampling rate in Hz (MANDATORY for .mat files)"
    )
    expected_fs: float | None = Field(
        default=None, description="Alternative key for sampling rate in Hz"
    )

    n_balls: int | None = Field(
        default=None, description="Number of rolling elements (bearing balls/rollers)"
    )
    ball_diameter: float | None = Field(
        default=None, description="Ball/roller diameter in mm"
    )
    pitch_diameter: float | None = Field(
        default=None, description="Pitch diameter in mm"
    )
    contact_angle_deg: float | None = Field(
        default=0.0, description="Bearing contact angle in degrees"
    )

    n_cylinders: int | None = Field(
        default=None, description="Number of cylinders (for reciprocating machines)"
    )
    stroke_type: Literal["2-stroke", "4-stroke"] | str | None = Field(
        default=None, description="Stroke cycle type"
    )

    primary_channel: str | None = Field(
        default=None, description="Preferred primary channel for analysis (e.g. 'DE', 'FE', 'BA', 'main')"
    )

    extra: dict[str, Any] | None = Field(
        default=None, description="Additional custom metadata tags"
    )


class DiagnosisRequest(BaseModel):
    query: str | None = Field(
        default=None, description="User diagnosis question. If omitted and a signal is provided, an auto-diagnostic query is generated."
    )
    signal_id: str | None = Field(
        default=None, description="Unique ID of previously uploaded signal file (from /signals/upload)"
    )
    signal_file_path: str | None = Field(
        default=None, description="Direct absolute or relative file path to the vibration signal on the server"
    )
    session_id: str | None = Field(
        default=None, description="Unique session/conversation thread identifier for multi-turn history"
    )
    machine_metadata: MachineMetadataDTO | None = Field(
        default=None, description="Machine operating parameters, bearing geometry, and ISO classification"
    )
    plot_format: PlotFormat = Field(
        default=PlotFormat.JSON, description="Output format for diagnostic figures ('json' for Plotly dict, 'html' for embeddable HTML)"
    )


class SignalAnalyzeRequest(BaseModel):
    signal_id: str | None = Field(
        default=None, description="Unique ID of previously uploaded signal"
    )
    signal_file_path: str | None = Field(
        default=None, description="Server file path to the vibration signal"
    )
    machine_metadata: MachineMetadataDTO | None = Field(
        default=None, description="Machine parameters and bearing geometry"
    )
    plot_format: PlotFormat = Field(
        default=PlotFormat.JSON, description="Output format for diagnostic plots"
    )


class RetrievalSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query for vibration analysis knowledge base")
    text_top_k: int = Field(default=2, ge=1, le=20, description="Number of text passages to retrieve")
    visual_top_k: int = Field(default=3, ge=1, le=20, description="Number of visual evidence items to retrieve")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="Reranker score threshold override")
    deduplicate_parents: bool = Field(default=True, description="Whether to deduplicate parent chunk contexts")


class DecomposeRequest(BaseModel):
    query: str = Field(..., description="Query to analyze for complexity and sub-query decomposition")
