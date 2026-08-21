"""
VibraDiag API — Sessions & History Management Router.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from src.api.dependencies import get_session_service
from src.api.schemas.responses import RecentSessionsListResponse, SessionStateResponse
from src.api.services.session_service import SessionService

router = APIRouter(prefix="/api/v1/sessions", tags=["Session & History Management"])


@router.get(
    "",
    response_model=RecentSessionsListResponse,
    summary="List Recent Sessions",
    description="Retrieves a list of recent conversation threads with metadata from the checkpointer.",
)
async def list_recent_sessions(
    session_svc: Annotated[SessionService, Depends(get_session_service)],
    limit: int = 5,
) -> RecentSessionsListResponse:
    summaries = session_svc.list_recent_sessions(limit=limit)
    return RecentSessionsListResponse(count=len(summaries), sessions=summaries)


@router.get(
    "/{session_id}",
    response_model=SessionStateResponse,
    summary="Get Session State Snapshot",
    description="Retrieves current checkpointed state for a conversation/diagnostic session.",
)
async def get_session(
    session_id: str,
    session_svc: Annotated[SessionService, Depends(get_session_service)],
) -> SessionStateResponse:
    return session_svc.get_session_state(session_id)


@router.get(
    "/{session_id}/messages",
    response_model=list[dict[str, Any]],
    summary="Get Session Message History",
    description="Returns list of conversational turns (Human/AI messages) for multi-turn followups.",
)
async def get_session_messages(
    session_id: str,
    session_svc: Annotated[SessionService, Depends(get_session_service)],
) -> list[dict[str, Any]]:
    return session_svc.get_session_messages(session_id)


@router.delete(
    "/{session_id}",
    summary="Delete Session Checkpoint",
    description="Clears session state and history from checkpointer storage.",
)
async def delete_session(
    session_id: str,
    session_svc: Annotated[SessionService, Depends(get_session_service)],
) -> dict[str, Any]:
    return session_svc.delete_session(session_id)
