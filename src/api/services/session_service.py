"""
VibraDiag API — Session & Checkpoint Management Service.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger

from config_loader import load_appcfg
from src.api.schemas.responses import SessionStateResponse


class AsyncCompatibleSqliteSaver(SqliteSaver):
    """
    SqliteSaver subclass that bridges async checkpointer calls (aput, aget_tuple, alist, aput_writes)
    to their synchronous counterparts to prevent 'NotImplementedError' in async FastAPI contexts.
    """

    async def aput(self, config, checkpoint, metadata, new_versions):
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id):
        return self.put_writes(config, writes, task_id)

    async def aget_tuple(self, config):
        return self.get_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def adelete_thread(self, thread_id: str):
        if hasattr(self, "delete_thread"):
            return self.delete_thread(thread_id)


class SessionService:
    """Manages conversation sessions and LangGraph checkpoint states."""

    def __init__(self, checkpointer_type: str | None = None, db_path: str | Path | None = None):
        app_cfg = load_appcfg()
        api_cfg = getattr(app_cfg, "api", {}) or {}

        c_type = (checkpointer_type or api_cfg.get("checkpointer_type", "sqlite")).lower()
        self.checkpointer_type = c_type
        self.checkpointer: BaseCheckpointSaver

        if c_type == "sqlite":
            target_path = Path(db_path or api_cfg.get("sqlite_db_path", "./data/sessions.db")).resolve()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_conn = sqlite3.connect(str(target_path), check_same_thread=False)
            self.checkpointer = AsyncCompatibleSqliteSaver(conn=self._sqlite_conn)
            # Setup tables if needed
            self.checkpointer.setup()
            logger.info(f"SessionService initialized with AsyncCompatibleSqliteSaver at '{target_path}'")
        else:
            self.checkpointer = MemorySaver()
            logger.info("SessionService initialized with MemorySaver")

    def get_checkpointer(self) -> BaseCheckpointSaver:
        return self.checkpointer

    def get_session_state(self, session_id: str) -> SessionStateResponse:
        """Retrieves checkpointed state snapshot for a given session ID."""
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state_tuple = self.checkpointer.get_tuple(config)

        if not state_tuple or not state_tuple.checkpoint:
            raise HTTPException(
                status_code=404,
                detail=f"Oturum bulunamadı veya henüz bir durum kaydedilmedi: '{session_id}'",
            )

        channel_values = state_tuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])

        serializable_messages = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                serializable_messages.append(msg.model_dump())
            elif hasattr(msg, "content"):
                role = getattr(msg, "type", "unknown")
                serializable_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                serializable_messages.append(msg)

        return SessionStateResponse(
            session_id=session_id,
            message_count=len(serializable_messages),
            route_decision=channel_values.get("route_decision"),
            primary_fault=channel_values.get("primary_fault"),
            messages=serializable_messages,
            errors=channel_values.get("errors", []),
        )

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieves conversational messages for multi-turn history."""
        session_data = self.get_session_state(session_id)
        return session_data.messages

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Clears/resets checkpoint state for a specific thread."""
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state_tuple = self.checkpointer.get_tuple(config)
        if not state_tuple or not state_tuple.checkpoint:
            raise HTTPException(status_code=404, detail=f"Silinecek oturum bulunamadı: '{session_id}'")

        try:
            if hasattr(self.checkpointer, "delete_thread"):
                self.checkpointer.delete_thread(thread_id=session_id)
            elif isinstance(self.checkpointer, MemorySaver) and hasattr(self.checkpointer, "storage"):
                self.checkpointer.storage.pop(session_id, None)
            logger.info(f"Deleted checkpoint state for session '{session_id}'")
            return {"status": "success", "message": f"Oturum '{session_id}' başarıyla temizlendi."}
        except Exception as e:
            logger.warning(f"Error during session deletion: {e}")
            return {"status": "success", "message": f"Oturum '{session_id}' sıfırlandı."}

    def list_recent_sessions(self, limit: int = 5) -> list[dict[str, Any]]:
        """Lists the most recently active sessions from SQLite checkpointer."""
        if not hasattr(self, "_sqlite_conn") or self._sqlite_conn is None:
            return []

        try:
            cursor = self._sqlite_conn.cursor()
            cursor.execute(
                """
                SELECT thread_id, count(*), max(checkpoint_id)
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY max(checkpoint_id) DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            summaries = []
            for thread_id, checkpoint_count, last_cp in rows:
                preview_text = None
                primary_fault = None
                msg_count = 0
                try:
                    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
                    state_tuple = self.checkpointer.get_tuple(config)
                    if state_tuple and state_tuple.checkpoint:
                        chan = state_tuple.checkpoint.get("channel_values", {})
                        primary_fault = chan.get("primary_fault")
                        msgs = chan.get("messages", [])
                        msg_count = len(msgs)
                        for m in msgs:
                            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                            if content:
                                preview_text = str(content)[:50]
                                break
                except Exception:
                    pass

                summaries.append(
                    {
                        "session_id": thread_id,
                        "message_count": msg_count,
                        "primary_fault": primary_fault,
                        "last_query_preview": preview_text,
                        "last_checkpoint_id": str(last_cp),
                    }
                )
            return summaries
        except Exception as e:
            logger.warning(f"Failed to query recent sessions from SQLite: {e}")
            return []
