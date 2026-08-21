"""
VibraDiag API — Routers Module Exports.
"""

from src.api.routers.config_routes import router as config_router
from src.api.routers.diagnosis import router as diagnosis_router
from src.api.routers.health import router as health_router
from src.api.routers.retrieval import router as retrieval_router
from src.api.routers.sessions import router as sessions_router
from src.api.routers.signals import router as signals_router

__all__ = [
    "config_router",
    "diagnosis_router",
    "health_router",
    "retrieval_router",
    "sessions_router",
    "signals_router",
]
