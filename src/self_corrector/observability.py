"""
VibraDiag — Self-Corrector Observability & Dual Telemetry.

Provides:
1. LangSmith Run Feedback & Tagging (Score tracking, cost & token monitoring, execution errors)
2. Local Event Stream (corrector_events.jsonl for offline analytics & Corpus Gap discovery)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from schemas.schemas import CorrectionDecision, TokenUsageInfo

logger = logging.getLogger(__name__)


class LocalCorrectorEventLogger:
    """
    Appends structured events to local corrector_events.jsonl file
    for offline SQL/DuckDB aggregation without putting load on LangSmith APIs.
    """

    @classmethod
    def get_log_path(cls) -> Path:
        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = project_root / "data" / "eval_reports"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "corrector_events.jsonl"

    @classmethod
    def log_event(
        cls,
        session_id: str | None,
        query: str,
        decision: CorrectionDecision,
        fault_type: str | None = None,
        machine_class: str | None = None,
        attempt_number: int = 1,
    ) -> None:
        try:
            execution_errors = {
                name: getattr(res, "execution_error", False)
                for name, res in decision.checker_results.items()
            }
            error_messages = {
                name: getattr(res, "error_message", None)
                for name, res in decision.checker_results.items()
                if getattr(res, "execution_error", False)
            }
            has_execution_error = any(execution_errors.values())

            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id or "unknown",
                "query": query,
                "attempt_number": attempt_number,
                "fault_type": fault_type or "unknown",
                "machine_class": machine_class or "unknown",
                "status": decision.status,
                "trigger_reasons": decision.trigger_reasons,
                "strategy_used": decision.strategy_used,
                "reformulated_query": decision.reformulated_query,
                "scores": {
                    name: res.score for name, res in decision.checker_results.items()
                },
                "passed": {
                    name: res.passed for name, res in decision.checker_results.items()
                },
                "execution_errors": execution_errors,
                "has_execution_error": has_execution_error,
                "error_messages": error_messages,
                "tokens": {
                    "prompt": decision.total_tokens.prompt_tokens,
                    "completion": decision.total_tokens.completion_tokens,
                    "total": decision.total_tokens.total_tokens,
                },
                "estimated_cost_usd": decision.total_tokens.estimated_cost_usd,
                "latency_ms": decision.total_latency_ms,
                "needs_review": decision.needs_review,
            }

            filepath = cls.get_log_path()
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

            logger.debug("Logged corrector event to %s", filepath)
        except Exception as err:
            logger.warning("Failed to write to local corrector event stream: %s", err)


class LangSmithCorrectorObserver:
    """
    Submits structured feedback and scores to LangSmith for online dashboard tracking.
    """

    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            try:
                from langsmith import Client
                api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
                if api_key:
                    cls._client = Client(api_key=api_key)
            except Exception as e:
                logger.debug("LangSmith client initialization skipped: %s", e)
        return cls._client

    @classmethod
    def record_decision_feedback(
        cls,
        run_id: str | None,
        decision: CorrectionDecision,
    ) -> None:
        client = cls.get_client()
        if not client or not run_id:
            return


        try:
            uuid.UUID(str(run_id))
        except (ValueError, TypeError):
            logger.debug("LangSmith feedback skipped: run_id '%s' is not a valid UUID", run_id)
            return

        try:
            for name, res in decision.checker_results.items():
                client.create_feedback(
                    run_id=run_id,
                    key=f"{name}_score",
                    score=res.score,
                    comment=res.rationale,
                )
                if getattr(res, "execution_error", False):
                    client.create_feedback(
                        run_id=run_id,
                        key=f"{name}_execution_error",
                        score=1.0,
                        comment=getattr(res, "error_message", "LLM Judge call failed"),
                    )

            client.create_feedback(
                run_id=run_id,
                key="correction_cost_usd",
                score=decision.total_tokens.estimated_cost_usd,
                comment=f"Tokens: {decision.total_tokens.total_tokens}",
            )

            client.create_feedback(
                run_id=run_id,
                key="correction_latency_ms",
                score=decision.total_latency_ms,
            )

            if decision.needs_review:
                client.create_feedback(
                    run_id=run_id,
                    key="flagged_low_confidence",
                    score=1.0,
                    comment=decision.warning_message or "Max correction attempts reached / Execution error",
                )

            logger.info("Successfully pushed corrector feedback to LangSmith for run %s", run_id)
        except Exception as err:
            logger.debug("LangSmith feedback submission skipped/failed: %s", err)
