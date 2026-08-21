"""
VibraDiag API — Digital Signal Processing & Diagnostic Schemas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TimeDomainStatsDTO(BaseModel):
    rms: float = Field(default=0.0, description="Root Mean Square vibration amplitude (mm/s)")
    peak: float = Field(default=0.0, description="Peak vibration amplitude")
    crest_factor: float = Field(default=0.0, description="Crest factor (Peak / RMS)")
    kurtosis: float = Field(default=0.0, description="Kurtosis (measure of signal impulsiveness)")
    skewness: float | None = Field(default=None, description="Skewness")
    margin_factor: float | None = Field(default=None, description="Margin factor")
    shape_factor: float | None = Field(default=None, description="Shape factor")


class ISOSeverityDTO(BaseModel):
    zone: str = Field(default="Unknown", description="ISO 2372 / 10816 Zone (A: Good, B: Satisfactory, C: Just Tolerable, D: Unacceptable/Danger)")
    meaning: str = Field(default="", description="Human-readable ISO zone meaning")
    rms_mm_s: float = Field(default=0.0, description="Evaluated RMS velocity in mm/s")
    machine_class: str | None = Field(default=None, description="Machine class (Class I, II, III, IV)")
    machine_description: str | None = Field(default=None, description="Detailed machine class description")
    range_mm_s: list[float] | None = Field(default=None, description="Applicable RMS velocity range for the zone")
    severity_table: dict[str, Any] | None = Field(default=None, description="Complete standard ISO lookup reference")


class DiagnosticPlotDTO(BaseModel):
    fft_spectrum: dict[str, Any] | str | None = Field(
        default=None, description="FFT Spectrum (Plotly JSON dict, HTML string, or file path)"
    )
    envelope_spectrum: dict[str, Any] | str | None = Field(
        default=None, description="Envelope Spectrum (Plotly JSON dict, HTML string, or file path)"
    )
    kurtogram: dict[str, Any] | str | None = Field(
        default=None, description="Spectral Kurtogram Heatmap (Plotly JSON dict, HTML string, or file path)"
    )
    fft_spectrum_path: str | None = Field(
        default=None, description="Standalone HTML file path for FFT Spectrum plot"
    )
    envelope_spectrum_path: str | None = Field(
        default=None, description="Standalone HTML file path for Envelope Spectrum plot"
    )
    kurtogram_path: str | None = Field(
        default=None, description="Standalone HTML file path for Spectral Kurtogram Heatmap plot"
    )
    orbit_plot: dict[str, Any] | str | None = Field(
        default=None, description="Polar Orbit / Phase Portrait Plot (Plotly JSON dict, HTML string, or file path)"
    )
    orbit_plot_path: str | None = Field(
        default=None, description="Standalone HTML file path for Polar Orbit plot"
    )
