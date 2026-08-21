"""
VibraDiag API — FastAPI Dependency Injection Providers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from config_loader import AppConfig, load_appcfg
from main_graph import get_default_retrieval_graph
from retrieval.retriever import RetrievalGraph
from src.api.services.graph_service import GraphService
from src.api.services.session_service import SessionService
from src.api.services.signal_service import SignalService

_signal_service_instance: SignalService | None = None
_session_service_instance: SessionService | None = None
_graph_service_instance: GraphService | None = None


def get_app_config() -> AppConfig:
    """Returns cached AppConfig instance."""
    return load_appcfg()


def get_signal_service() -> SignalService:
    """Returns singleton SignalService instance."""
    global _signal_service_instance
    if _signal_service_instance is None:
        _signal_service_instance = SignalService()
    return _signal_service_instance


def get_session_service() -> SessionService:
    """Returns singleton SessionService instance."""
    global _session_service_instance
    if _session_service_instance is None:
        _session_service_instance = SessionService()
    return _session_service_instance


def get_graph_service(
    signal_svc: Annotated[SignalService, Depends(get_signal_service)],
    session_svc: Annotated[SessionService, Depends(get_session_service)],
) -> GraphService:
    """Returns singleton GraphService instance."""
    global _graph_service_instance
    if _graph_service_instance is None:
        _graph_service_instance = GraphService(
            signal_service=signal_svc,
            session_service=session_svc,
        )
    return _graph_service_instance


def get_retrieval_graph() -> RetrievalGraph:
    """Returns singleton RetrievalGraph instance."""
    return get_default_retrieval_graph()
