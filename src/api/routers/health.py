"""
VibraDiag API — Health & Readiness Checks Router.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from config_loader import AppConfig
from src.api.dependencies import get_app_config
from src.api.schemas.responses import HealthResponse

router = APIRouter(tags=["Health & Readiness"])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Returns OK if the API server is up and accepting incoming HTTP connections.",
)
async def liveness_check(
    app_cfg: Annotated[AppConfig, Depends(get_app_config)],
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=app_cfg.app.get("name", "VibraDiag"),
        version=app_cfg.app.get("version", "0.1.0"),
        timestamp=datetime.now(UTC).isoformat(),
        components={"server": "healthy"},
    )


@router.get(
    "/readyz",
    summary="Readiness Probe",
    description="Validates connections to Qdrant, DocStore SQLite, and LLM API keys.",
)
async def readiness_check(
    app_cfg: Annotated[AppConfig, Depends(get_app_config)],
) -> JSONResponse:
    checks: dict[str, Any] = {}
    is_ready = True

    docstore_path = Path(app_cfg.paths.get("docstore_path", "./docstore.db")).resolve()
    if docstore_path.exists():
        checks["docstore"] = {"status": "ok", "path": str(docstore_path)}
    else:
        checks["docstore"] = {"status": "warning", "message": f"Docstore not found at {docstore_path}"}

    qdrant_path = Path(app_cfg.paths.get("qdrant_persist_path", "./qdrant_data")).resolve()
    if qdrant_path.exists():
        checks["qdrant"] = {"status": "ok", "path": str(qdrant_path)}
    else:
        checks["qdrant"] = {"status": "warning", "message": f"Qdrant data not found at {qdrant_path}"}

    groq_key = bool(os.getenv("GROQ_API_KEY"))
    gemini_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    checks["groq_api"] = {"configured": groq_key}
    checks["gemini_api"] = {"configured": gemini_key}

    if not groq_key and not gemini_key:
        is_ready = False

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": checks,
        },
    )
