"""
VibraDiag API — Signal Management & DSP Service.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from fastapi import HTTPException, UploadFile
from loguru import logger

from config_loader import load_appcfg
from deterministic_tools.fault_analyzer import pick_primary_fault
from deterministic_tools.reader import LoadedSignal, SignalReaderFactory
from deterministic_tools.signal_processing_subgraph import (
    plot_diagnostics_node,
    signal_processing_subgraph_node,
)
from src.api.schemas.common import PlotFormat
from src.api.schemas.dsp_models import DiagnosticPlotDTO, ISOSeverityDTO, TimeDomainStatsDTO
from src.api.schemas.requests import MachineMetadataDTO
from src.api.schemas.responses import SignalAnalysisResponse, SignalUploadResponse


class SignalService:
    """Service handling signal uploads, validation, storage and isolated DSP operations."""

    def __init__(self, upload_dir: str | Path | None = None, max_size_mb: int = 50):
        app_cfg = load_appcfg()
        api_cfg = getattr(app_cfg, "api", {}) or {}

        target_dir = upload_dir or api_cfg.get("upload_dir", "./data/uploads")
        self.upload_dir = Path(target_dir).resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = (api_cfg.get("max_upload_size_mb") or max_size_mb) * 1024 * 1024
        self._signal_registry: dict[str, Path] = {}

    def get_signal_path(self, signal_id_or_path: str) -> Path:
        """Resolves signal ID from memory registry or treats as a direct file path."""
        if signal_id_or_path in self._signal_registry:
            path = self._signal_registry[signal_id_or_path]
            if path.exists():
                return path

        direct_path = Path(signal_id_or_path).resolve()
        if direct_path.exists() and direct_path.is_file():
            return direct_path

        in_upload = (self.upload_dir / signal_id_or_path).resolve()
        if in_upload.exists() and in_upload.is_file():
            return in_upload

        raise HTTPException(
            status_code=404,
            detail=f"Sinyal dosyası bulunamadı: '{signal_id_or_path}'",
        )

    async def save_uploaded_file(
        self,
        file: UploadFile,
        fs: float | None = None,
        rpm: float | None = None,
    ) -> SignalUploadResponse:
        """Saves uploaded file to disk and extracts initial metadata."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="Dosya adı boş olamaz.")

        ext = Path(file.filename).suffix.lower()
        if ext not in (".mat", ".wav", ".csv"):
            raise HTTPException(
                status_code=400,
                detail=f"Desteklenmeyen dosya formatı: '{ext}'. Yalnızca .mat, .wav ve .csv desteklenir.",
            )

        signal_id = f"sig_{uuid.uuid4().hex[:12]}"
        dest_filename = f"{signal_id}_{file.filename}"
        dest_path = self.upload_dir / dest_filename

        total_bytes = 0
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > self.max_size_bytes:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Dosya boyutu izin verilen sınırı aşıyor (Maks: {self.max_size_bytes // (1024*1024)} MB).",
                    )
                buffer.write(chunk)

        self._signal_registry[signal_id] = dest_path

        channels: list[str] = []
        detected_fs: float | None = fs
        detected_rpm: float | None = rpm
        duration: float | None = None

        try:
            if ext == ".mat" and fs is None:
                loaded = None
            else:
                loaded = SignalReaderFactory.load(str(dest_path), expected_fs=fs)
                channels = list(loaded.channels.keys())
                detected_fs = loaded.fs
                if loaded.rpm and detected_rpm is None:
                    detected_rpm = float(loaded.rpm)
                if channels and detected_fs and detected_fs > 0:
                    first_ch = next(iter(loaded.channels.values()))
                    duration = round(len(first_ch) / detected_fs, 3)
        except Exception as e:
            logger.warning(f"Could not pre-read uploaded signal metadata: {e}")

        return SignalUploadResponse(
            signal_id=signal_id,
            filename=file.filename,
            file_path=str(dest_path),
            size_bytes=total_bytes,
            format=ext,
            channels=channels,
            fs=detected_fs,
            rpm=detected_rpm,
            duration_seconds=duration,
            message="Sinyal dosyası başarıyla yüklendi." if (ext != ".mat" or fs is not None) else "Sinyal yüklendi. (.mat formatı için analiz aşamasında 'fs' belirtilmelidir.)",
        )

    def validate_and_load_signal(
        self,
        signal_file_path: Path | str,
        meta_dto: MachineMetadataDTO | None = None,
    ) -> tuple[LoadedSignal, dict[str, Any]]:
        """
        Validates signal parameters and reads signal into LoadedSignal.
        Enforces strict 'fs' validation for .mat files.
        """
        path = Path(signal_file_path).resolve()
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Sinyal dosyası bulunamadı: {path}")

        meta_dict = meta_dto.model_dump(exclude_none=True) if meta_dto else {}
        expected_fs = meta_dict.get("fs") or meta_dict.get("expected_fs")

        if path.suffix.lower() == ".mat" and (expected_fs is None or float(expected_fs) <= 0):
            raise HTTPException(
                status_code=422,
                detail=".mat formatındaki titreşim sinyalleri için 'fs' (örnekleme frekansı, örn: 12000) parametresi zorunludur.",
            )

        try:
            loaded = SignalReaderFactory.load(str(path), expected_fs=expected_fs)
            if loaded.rpm and float(loaded.rpm) > 0:
                if "rpm" not in meta_dict or meta_dict.get("rpm") == 1797.0:
                    meta_dict["rpm"] = float(loaded.rpm)
            if loaded.fs and "fs" not in meta_dict:
                meta_dict["fs"] = float(loaded.fs)
            return loaded, meta_dict
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Sinyal parametre hatası: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load signal '{path}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Sinyal okunamadı: {e}") from e

    def run_isolated_dsp(
        self,
        signal_file_path: Path | str,
        meta_dto: MachineMetadataDTO | None = None,
        plot_format: PlotFormat = PlotFormat.JSON,
    ) -> SignalAnalysisResponse:
        """Runs only the deterministic SignalProcessingSubGraph (0 LLM token cost)."""
        path = self.get_signal_path(str(signal_file_path))
        loaded, sanitized_meta = self.validate_and_load_signal(path, meta_dto)

        parent_state = {
            "signal_file_path": str(path),
            "loaded_signal": loaded,
            "machine_metadata": sanitized_meta,
            "errors": [],
        }

        dsp_output = signal_processing_subgraph_node(parent_state)
        sig_res = dsp_output.get("signal_processing_result", {})

        fault_summary = pick_primary_fault(
            direct_spectrum_diagnosis=sig_res.get("direct_spectrum_diagnosis"),
            envelope_diagnosis=sig_res.get("envelope_diagnosis"),
            reciprocating_diagnosis=sig_res.get("reciprocating_diagnosis"),
        )
        primary_fault = fault_summary.get("primary_fault_name")

        td_stats_dict = sig_res.get("time_domain_stats") or {}
        rms_val = td_stats_dict.get("rms") or td_stats_dict.get("overall_rms") or td_stats_dict.get("acceleration_rms") or 0.0
        peak_val = td_stats_dict.get("peak") or td_stats_dict.get("peak_value") or 0.0
        time_domain_dto = TimeDomainStatsDTO(
            rms=float(rms_val),
            peak=float(peak_val),
            crest_factor=float(td_stats_dict.get("crest_factor", 0.0)),
            kurtosis=float(td_stats_dict.get("kurtosis", 0.0)),
            skewness=td_stats_dict.get("skewness"),
            margin_factor=td_stats_dict.get("margin_factor"),
            shape_factor=td_stats_dict.get("shape_factor"),
        )

        iso_dto = None
        severity_alert = None
        iso_zone = sig_res.get("iso_severity_zone")
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
            if iso_zone.upper() == "D":
                severity_alert = "EMERGENCY_STOP_RECOMMENDED (ISO Zone D: Unacceptable / Dangerous Vibration)"

        parent_state["signal_processing_result"] = sig_res
        plots_dict = dsp_output.get("diagnostic_plots")
        if not plots_dict:
            plots_output = plot_diagnostics_node(parent_state)
            plots_dict = plots_output.get("diagnostic_plots", {})

        formatted_plots = self._format_plots(plots_dict, plot_format)

        is_compound = bool(dsp_output.get("is_compound_fault", False) or fault_summary.get("is_compound_fault", False))
        secondary_name = dsp_output.get("secondary_fault_name") or fault_summary.get("secondary_fault_name")
        secondary_abbr = dsp_output.get("secondary_fault_abbr") or fault_summary.get("secondary_fault_abbr")
        co_occurring = dsp_output.get("co_occurring_faults") or fault_summary.get("co_occurring_faults") or []
        detected_rpm = dsp_output.get("detected_rpm") or sig_res.get("rpm")

        return SignalAnalysisResponse(
            signal_id=str(signal_file_path) if str(signal_file_path) in self._signal_registry else None,
            primary_fault=primary_fault,
            primary_fault_data=fault_summary,
            is_compound_fault=is_compound,
            secondary_fault_name=secondary_name,
            secondary_fault_abbr=secondary_abbr,
            co_occurring_faults=co_occurring,
            detected_rpm=float(detected_rpm) if detected_rpm else None,
            iso_severity=iso_dto,
            severity_alert=severity_alert,
            time_domain_stats=time_domain_dto,
            dsp_analysis=sig_res,
            diagnostic_plots=formatted_plots,
        )

    def _format_plots(
        self,
        plots_dict: dict[str, Any],
        plot_format: PlotFormat,
    ) -> DiagnosticPlotDTO:
        """Converts plot state dictionary into DiagnosticPlotDTO supporting paths, HTML, and figures."""
        if not plots_dict:
            return DiagnosticPlotDTO()

        fft_plot = plots_dict.get("fft_spectrum_dict") or plots_dict.get("fft_spectrum")
        env_plot = plots_dict.get("envelope_spectrum_dict") or plots_dict.get("envelope_spectrum")
        kurt_plot = plots_dict.get("kurtogram_dict") or plots_dict.get("kurtogram")
        orbit_plot = plots_dict.get("orbit_plot_dict") or plots_dict.get("orbit_plot")
        fft_path = plots_dict.get("fft_spectrum_path") or (fft_plot if isinstance(fft_plot, str) and fft_plot.endswith(".html") else None)
        env_path = plots_dict.get("envelope_spectrum_path") or (env_plot if isinstance(env_plot, str) and env_plot.endswith(".html") else None)
        kurt_path = plots_dict.get("kurtogram_path") or (kurt_plot if isinstance(kurt_plot, str) and kurt_plot.endswith(".html") else None)
        orbit_path = plots_dict.get("orbit_plot_path") or (orbit_plot if isinstance(orbit_plot, str) and orbit_plot.endswith(".html") else None)

        if plot_format == PlotFormat.HTML:
            fft_html = None
            env_html = None
            kurt_html = None
            orbit_html = None
            if isinstance(fft_plot, dict):
                try:
                    fig = go.Figure(fft_plot)
                    fft_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                except Exception as e:
                    logger.warning(f"Failed to render FFT plot to HTML: {e}")
            elif isinstance(fft_plot, str):
                if Path(fft_plot).exists() and fft_plot.endswith(".html"):
                    try:
                        fft_html = Path(fft_plot).read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to read plot HTML file '{fft_plot}': {e}")
                        fft_html = fft_plot
                else:
                    fft_html = fft_plot

            if isinstance(env_plot, dict):
                try:
                    fig = go.Figure(env_plot)
                    env_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                except Exception as e:
                    logger.warning(f"Failed to render Envelope plot to HTML: {e}")
            elif isinstance(env_plot, str):
                if Path(env_plot).exists() and env_plot.endswith(".html"):
                    try:
                        env_html = Path(env_plot).read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to read plot HTML file '{env_plot}': {e}")
                        env_html = env_plot
                else:
                    env_html = env_plot

            if isinstance(kurt_plot, dict):
                try:
                    fig = go.Figure(kurt_plot)
                    kurt_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                except Exception as e:
                    logger.warning(f"Failed to render Kurtogram to HTML: {e}")
            elif isinstance(kurt_plot, str):
                if Path(kurt_plot).exists() and kurt_plot.endswith(".html"):
                    try:
                        kurt_html = Path(kurt_plot).read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to read plot HTML file '{kurt_plot}': {e}")
                        kurt_html = kurt_plot
                else:
                    kurt_html = kurt_plot

            if isinstance(orbit_plot, dict):
                try:
                    fig = go.Figure(orbit_plot)
                    orbit_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                except Exception as e:
                    logger.warning(f"Failed to render Orbit plot to HTML: {e}")
            elif isinstance(orbit_plot, str):
                if Path(orbit_plot).exists() and orbit_plot.endswith(".html"):
                    try:
                        orbit_html = Path(orbit_plot).read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to read plot HTML file '{orbit_plot}': {e}")
                        orbit_html = orbit_plot
                else:
                    orbit_html = orbit_plot

            return DiagnosticPlotDTO(
                fft_spectrum=fft_html,
                envelope_spectrum=env_html,
                kurtogram=kurt_html,
                orbit_plot=orbit_html,
                fft_spectrum_path=fft_path,
                envelope_spectrum_path=env_path,
                kurtogram_path=kurt_path,
                orbit_plot_path=orbit_path,
            )

        return DiagnosticPlotDTO(
            fft_spectrum=fft_plot,
            envelope_spectrum=env_plot,
            kurtogram=kurt_plot,
            orbit_plot=orbit_plot,
            fft_spectrum_path=fft_path,
            envelope_spectrum_path=env_path,
            kurtogram_path=kurt_path,
            orbit_plot_path=orbit_path,
        )
