"""
VibraDiag API — Configuration & Telemetry Router.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from config_loader import AppConfig, load_retcfg
from src.api.dependencies import get_app_config

router = APIRouter(prefix="/api/v1/config", tags=["Configuration & Telemetry"])


@router.get(
    "/thresholds",
    summary="Get Active Diagnostic Thresholds",
    description="Returns current threshold values for self-corrector, reranker, and query complexity classifier.",
)
async def get_thresholds(
    app_cfg: Annotated[AppConfig, Depends(get_app_config)],
) -> dict[str, Any]:
    ret_cfg = load_retcfg()
    return {
        "self_corrector": app_cfg.self_corrector,
        "reranker": ret_cfg.hybrid_search.get("reranker", {}),
        "decomposer": app_cfg.decomposer,
    }


@router.get(
    "/info",
    summary="Get System Configuration Info",
    description="Returns sanitized system settings, model names, and storage paths.",
)
async def get_system_info(
    app_cfg: Annotated[AppConfig, Depends(get_app_config)],
) -> dict[str, Any]:
    return {
        "app": app_cfg.app,
        "api": app_cfg.api,
        "llm_model": app_cfg.llm.get("model"),
        "embeddings_model": app_cfg.embeddings.get("model"),
        "vision_model": app_cfg.vision_llm.get("model"),
        "paths": app_cfg.paths,
    }
