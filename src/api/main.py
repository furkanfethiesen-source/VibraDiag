"""
VibraDiag API — Main FastAPI Application Entry Point.
run => uv run uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from config_loader import load_appcfg
from src.api.middleware.error_handler import setup_exception_handlers
from src.api.middleware.logging_middleware import RequestLoggingMiddleware
from src.api.routers import (
    config_router,
    diagnosis_router,
    health_router,
    retrieval_router,
    sessions_router,
    signals_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and graceful shutdown."""
    app_cfg = load_appcfg()
    api_cfg = getattr(app_cfg, "api", {}) or {}

    upload_dir = Path(api_cfg.get("upload_dir", "./data/uploads")).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    import os
    langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2") or os.getenv("LANGSMITH_TRACING")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT", "VibraDiag")
    has_langsmith_key = bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))

    logger.info("=" * 60)
    logger.info(f"Starting {app.title} v{app.version}")
    logger.info(f"Upload directory: {upload_dir}")
    logger.info(f"Checkpointer: {api_cfg.get('checkpointer_type', 'sqlite')}")
    logger.info(f"LangSmith Tracing: {langsmith_tracing == 'true' and has_langsmith_key} (Project: {langsmith_project})")
    logger.info("=" * 60)

    yield

    logger.info(f"Shutting down {app.title}...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app_cfg = load_appcfg()
    api_cfg = getattr(app_cfg, "api", {}) or {}

    title = api_cfg.get("title", "VibraDiag Fault Diagnosis API")
    version = api_cfg.get("version", "0.1.0")
    description = api_cfg.get(
        "description",
        "Vibration Analysis, Signal Processing & Fault Diagnosis REST and SSE API",
    )

    app = FastAPI(
        title=title,
        version=version,
        description=description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    cors_origins = api_cfg.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestLoggingMiddleware)

    setup_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(diagnosis_router)
    app.include_router(signals_router)
    app.include_router(retrieval_router)
    app.include_router(sessions_router)
    app.include_router(config_router)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


app = create_app()


if __name__ == "__main__":
    app_cfg = load_appcfg()
    api_cfg = getattr(app_cfg, "api", {}) or {}

    host = api_cfg.get("host", "0.0.0.0")
    port = int(api_cfg.get("port", 8000))
    reload = bool(api_cfg.get("reload", True))

    logger.info(f"Launching Uvicorn server on http://{host}:{port} (reload={reload})")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )
