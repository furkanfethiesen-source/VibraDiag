"""
VibraDiag — Self-Corrector LangGraph Node.

Orchestrates:
1. Attempt & global cycle cap enforcement.
2. Fast-fail deterministic DSP validation (zero token cost) with negation handling.
3. Faithfulness (hallucination) & Relevance verification via Gemini 3.1 Flash Lite / Groq.
4. Strategic query reformulation (Dictionary -> PRF -> LLM Rewrite).
5. Observability telemetry (LangSmith Feedback + Local JSONL event stream).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from langsmith import traceable

from config_loader import load_appcfg
from generation.client import GroqClient
from schemas.schemas import (
    CheckerResult,
    CorrectionDecision,
    TokenUsageInfo,
)
from self_corrector.checkers import (
    DSPConsistencyChecker,
    FaithfulnessChecker,
    GeminiJudgeClient,
    RelevanceChecker,
)
from self_corrector.observability import (
    LangSmithCorrectorObserver,
    LocalCorrectorEventLogger,
)
from self_corrector.strategies import (
    DomainDictionaryExpansion,
    PseudoRelevanceFeedback,
    QueryRewriter,
)

logger = logging.getLogger(__name__)

_default_groq_client: GroqClient | None = None
_default_judge_client: Any | None = None


def get_default_groq_client() -> GroqClient:
    global _default_groq_client
    if _default_groq_client is None:
        _default_groq_client = GroqClient()
    return _default_groq_client


def get_default_judge_client() -> Any:
    global _default_judge_client
    if _default_judge_client is None:
        _default_judge_client = GeminiJudgeClient()
    return _default_judge_client


def load_thresholds() -> dict[str, Any]:
    default_config = {
        "dsp_consistency_threshold": 1.0,
        "faithfulness_threshold": 0.70,
        "relevance_threshold": 0.65,
        "prf_min_score_guard": 0.40,
        "max_retrieval_attempts": 2,
        "max_generation_attempts": 2,
        "global_max_cycles": 3,
    }
    try:
        app_cfg = load_appcfg()
        if hasattr(app_cfg, "self_corrector") and app_cfg.self_corrector:
            default_config.update(app_cfg.self_corrector)
    except Exception as e:
        logger.warning("Failed to load self_corrector config from app.yaml: %s", e)
    return default_config


@traceable(name="self_corrector_node")
def self_corrector_node(
    state: Any,
    thresholds_override: dict[str, Any] | None = None,
    judge_client: Any | None = None,
    groq_client: GroqClient | None = None,
) -> dict[str, Any]:
    """
    LangGraph self_corrector node.

    Reads from state:
    - llm_response: str
    - user_query: str
    - signal_analysis / signal_processing_result: dict (optional)
    - text_passages / merged_context / retrieval_results: list | str
    - sub_queries: list[str] (optional)
    - correction_attempts: dict[str, int]
    - excluded_chunk_ids: list[str]
    - max_text_score: float

    Writes to state:
    - route_decision: str ("approved" | "retry_retrieval" | "retry_generation" | "approved_with_warning")
    - reformulated_query: str | None
    - excluded_chunk_ids: list[str]
    - previous_failure_reason: str | None
    - is_flagged: bool
    - flag_reason: str | None
    - checker_results: dict[str, dict]
    - corrector_metrics: dict[str, Any]
    - corrector_history: list[dict]
    - llm_response: str (modified with warning if flagged)
    """
    start_total_time = time.time()

    if hasattr(state, "model_dump"):
        state_dict = state.model_dump()
    elif isinstance(state, dict):
        state_dict = state
    else:
        state_dict = dict(state)

    thresholds = load_thresholds()
    if thresholds_override:
        thresholds.update(thresholds_override)

    max_retrieval = int(thresholds.get("max_retrieval_attempts", 2))
    max_generation = int(thresholds.get("max_generation_attempts", 2))
    global_max = int(thresholds.get("global_max_cycles", 3))

    user_query = state_dict.get("user_query", "").strip()
    llm_response = state_dict.get("llm_response", "").strip()
    session_id = state_dict.get("session_id")
    sub_queries = state_dict.get("sub_queries")

    signal_data = (
        state_dict.get("signal_analysis")
        or state_dict.get("signal_processing_result")
        or state_dict.get("dsp_ground_truth")
    )

    context_chunks = state_dict.get("text_passages", [])
    if not context_chunks:
        merged_context = state_dict.get("merged_context", "")
        if merged_context:
            context_chunks = [{"id": "merged", "text": merged_context}]
        else:
            context_chunks = state_dict.get("retrieval_results", [])

    current_chunk_ids = []
    if isinstance(context_chunks, list):
        for c in context_chunks:
            cid = c.get("id") or c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", None)
            if cid:
                current_chunk_ids.append(cid)

    current_attempts = state_dict.get("correction_attempts") or {"retrieval": 0, "generation": 0, "total": 0}
    retrieval_attempts = current_attempts.get("retrieval", 0)
    generation_attempts = current_attempts.get("generation", 0)
    total_attempts = current_attempts.get("total", 0)

    existing_metrics = state_dict.get("corrector_metrics") or {
        "tokens_input": 0,
        "tokens_output": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "llm_calls": 0,
    }

    if total_attempts >= global_max or (retrieval_attempts >= max_retrieval and generation_attempts >= max_generation):
        warning_msg = (
            "Düzeltme döngüsü maksimum deneme sınırına ulaştı (Sistem güvenli çıkış yaptı)."
        )
        logger.warning("Self-corrector max attempts reached (total=%d). Flagging response.", total_attempts)

        flagged_response = f"[⚠️ Düşük Güven Uyarısı: Bu yanıt maksimum doğrulama sınırına ulaşmıştır.]\n\n{llm_response}"
        decision = CorrectionDecision(
            status="flagged",
            trigger_reasons=["max_attempts_exceeded"],
            needs_review=True,
            warning_message=warning_msg,
            total_latency_ms=(time.time() - start_total_time) * 1000,
        )

        LocalCorrectorEventLogger.log_event(
            session_id=session_id,
            query=user_query,
            decision=decision,
            attempt_number=total_attempts + 1,
        )

        return {
            "route_decision": "approved_with_warning",
            "is_flagged": True,
            "flag_reason": warning_msg,
            "llm_response": flagged_response,
            "correction_attempts": current_attempts,
            "checker_results": {},
            "corrector_history": [{
                "cycle": total_attempts + 1,
                "decision": "approved_with_warning",
                "reason": warning_msg,
            }],
        }

    errors = state_dict.get("errors") or []
    generation_failed = any("LLM Generation Error" in str(e) for e in errors[-1:])
    if not generation_failed and ("teknik hata oluştu" in llm_response.lower() or not llm_response.strip()):
        generation_failed = True

    if generation_failed:
        logger.warning("Upstream generation failure detected. Routing directly to retry_generation shortcut.")
        new_generation_attempts = generation_attempts + 1
        new_total_attempts = total_attempts + 1
        updated_attempts = {
            "retrieval": retrieval_attempts,
            "generation": new_generation_attempts,
            "total": new_total_attempts,
        }

        fail_reason = "LLM yanıt üretimi sırasında teknik hata oluştu."
        decision = CorrectionDecision(
            status="retry_generation",
            trigger_reasons=["generation_runtime_error"],
            checker_results={},
            total_tokens=TokenUsageInfo(),
            total_latency_ms=(time.time() - start_total_time) * 1000,
            needs_review=False,
        )

        LocalCorrectorEventLogger.log_event(
            session_id=session_id,
            query=user_query,
            decision=decision,
            attempt_number=new_total_attempts,
        )

        return {
            "route_decision": "retry_generation",
            "is_flagged": False,
            "flag_reason": None,
            "previous_failure_reason": fail_reason,
            "correction_attempts": updated_attempts,
            "checker_results": {},
            "corrector_history": [{
                "cycle": new_total_attempts,
                "decision": "retry_generation",
                "reason": fail_reason,
            }],
        }

    dsp_result: CheckerResult = DSPConsistencyChecker.check(
        generated_text=llm_response,
        signal_data=signal_data,
        threshold=thresholds.get("dsp_consistency_threshold", 1.0),
    )

    checker_results: dict[str, CheckerResult] = {"dsp_consistency": dsp_result}
    total_tokens = TokenUsageInfo()

    if not dsp_result.passed:
        logger.info("Fast-fail: DSP check failed. Routing directly to Generator.")
        new_generation_attempts = generation_attempts + 1
        new_total_attempts = total_attempts + 1
        updated_attempts = {
            "retrieval": retrieval_attempts,
            "generation": new_generation_attempts,
            "total": new_total_attempts,
        }

        total_latency = (time.time() - start_total_time) * 1000
        decision = CorrectionDecision(
            status="retry_generation",
            trigger_reasons=["dsp_consistency_mismatch"],
            checker_results=checker_results,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            needs_review=False,
        )

        LocalCorrectorEventLogger.log_event(
            session_id=session_id,
            query=user_query,
            decision=decision,
            attempt_number=new_total_attempts,
        )

        return {
            "route_decision": "retry_generation",
            "is_flagged": False,
            "flag_reason": None,
            "previous_failure_reason": dsp_result.rationale,
            "correction_attempts": updated_attempts,
            "checker_results": {k: v.model_dump() for k, v in checker_results.items()},
            "corrector_history": [{
                "cycle": new_total_attempts,
                "decision": "retry_generation",
                "reason": dsp_result.rationale,
            }],
        }

    active_judge_client = judge_client or get_default_judge_client()
    active_groq_client = groq_client or get_default_groq_client()

    faith_checker = FaithfulnessChecker(judge_client=active_judge_client)
    relevance_checker = RelevanceChecker(judge_client=active_judge_client)

    faith_result: CheckerResult = faith_checker.check(
        generated_text=llm_response,
        context_chunks=context_chunks,
        threshold=thresholds.get("faithfulness_threshold", 0.70),
    )
    checker_results["faithfulness"] = faith_result

    rel_result: CheckerResult = relevance_checker.check(
        user_query=user_query,
        generated_text=llm_response,
        sub_queries=sub_queries,
        threshold=thresholds.get("relevance_threshold", 0.65),
    )
    checker_results["sub_query_relevance"] = rel_result

    prompt_tokens = faith_result.token_usage.prompt_tokens + rel_result.token_usage.prompt_tokens
    comp_tokens = faith_result.token_usage.completion_tokens + rel_result.token_usage.completion_tokens
    cost_usd = faith_result.token_usage.estimated_cost_usd + rel_result.token_usage.estimated_cost_usd

    total_tokens = TokenUsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=comp_tokens,
        total_tokens=prompt_tokens + comp_tokens,
        estimated_cost_usd=round(cost_usd, 6),
    )

    failing_checkers: list[str] = [
        name for name, res in checker_results.items() if not res.passed
    ]

    has_execution_error = any(res.execution_error for res in checker_results.values())
    if has_execution_error:
        logger.warning("⚠️ Self-Corrector executed with fail-open fallback due to LLM error.")

    total_latency = (time.time() - start_total_time) * 1000

    if not failing_checkers:
        logger.info("✅ All self-corrector checks passed. Output approved.")
        decision = CorrectionDecision(
            status="approved",
            trigger_reasons=[],
            checker_results=checker_results,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            needs_review=has_execution_error,
        )

        LocalCorrectorEventLogger.log_event(
            session_id=session_id,
            query=user_query,
            decision=decision,
            attempt_number=total_attempts + 1,
        )

        failed_checkers_names = [k for k, res in checker_results.items() if getattr(res, "execution_error", False)]
        flag_reason = f"LLM judge execution error (fail-open) in: {', '.join(failed_checkers_names)}" if has_execution_error else None

        return {
            "route_decision": "approved",
            "is_flagged": has_execution_error,
            "flag_reason": flag_reason,
            "previous_failure_reason": None,
            "correction_attempts": current_attempts,
            "checker_results": {k: v.model_dump() for k, v in checker_results.items()},
            "corrector_metrics": {
                "tokens_input": existing_metrics["tokens_input"] + total_tokens.prompt_tokens,
                "tokens_output": existing_metrics["tokens_output"] + total_tokens.completion_tokens,
                "total_tokens": existing_metrics["total_tokens"] + total_tokens.total_tokens,
                "estimated_cost_usd": existing_metrics["estimated_cost_usd"] + total_tokens.estimated_cost_usd,
                "llm_calls": existing_metrics["llm_calls"] + 2,
            },
        }

    if "faithfulness" in failing_checkers or "sub_query_relevance" in failing_checkers:
        failure_reasons = [checker_results[c].rationale for c in failing_checkers]
        combined_reason = " | ".join(failure_reasons)
        strategy_used = "domain_dictionary"
        reformulated = user_query

        if retrieval_attempts == 0:
            strategy_used = "domain_dictionary"
            reformulated = DomainDictionaryExpansion.expand(user_query)
        elif retrieval_attempts == 1:
            strategy_used = "pseudo_relevance_feedback"
            max_score = state_dict.get("max_text_score", 0.0)
            prf_guard = thresholds.get("prf_min_score_guard", 0.40)
            reformulated = PseudoRelevanceFeedback.expand_with_prf(
                query=user_query,
                chunks=context_chunks if isinstance(context_chunks, list) else [],
                max_score=max_score,
                min_score_guard=prf_guard,
            )

            if reformulated == user_query:
                strategy_used = "llm_query_rewriter"
                rewriter = QueryRewriter(groq_client=active_groq_client)
                reformulated, rw_usage, _ = rewriter.rewrite(user_query, combined_reason)
                total_tokens.prompt_tokens += rw_usage.prompt_tokens
                total_tokens.completion_tokens += rw_usage.completion_tokens
                total_tokens.total_tokens += rw_usage.total_tokens
                total_tokens.estimated_cost_usd += rw_usage.estimated_cost_usd
        else:
            strategy_used = "llm_query_rewriter"
            rewriter = QueryRewriter(groq_client=active_groq_client)
            reformulated, rw_usage, _ = rewriter.rewrite(user_query, combined_reason)
            total_tokens.prompt_tokens += rw_usage.prompt_tokens
            total_tokens.completion_tokens += rw_usage.completion_tokens
            total_tokens.total_tokens += rw_usage.total_tokens
            total_tokens.estimated_cost_usd += rw_usage.estimated_cost_usd

        new_retrieval_attempts = retrieval_attempts + 1
        new_total_attempts = total_attempts + 1
        updated_attempts = {
            "retrieval": new_retrieval_attempts,
            "generation": generation_attempts,
            "total": new_total_attempts,
        }

        decision = CorrectionDecision(
            status="retry_retrieval",
            trigger_reasons=failing_checkers,
            checker_results=checker_results,
            reformulated_query=reformulated,
            strategy_used=strategy_used,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            needs_review=has_execution_error,
        )

        LocalCorrectorEventLogger.log_event(
            session_id=session_id,
            query=user_query,
            decision=decision,
            attempt_number=new_total_attempts,
        )

        new_excluded = current_chunk_ids
        
        failed_checkers_names = [k for k, res in checker_results.items() if getattr(res, "execution_error", False)]
        flag_reason = f"LLM judge execution error (fail-open) in: {', '.join(failed_checkers_names)}" if has_execution_error else None

        return {
            "route_decision": "retry_retrieval",
            "is_flagged": has_execution_error,
            "flag_reason": flag_reason,
            "reformulated_query": reformulated,
            "excluded_chunk_ids": new_excluded,
            "previous_failure_reason": combined_reason,
            "correction_attempts": updated_attempts,
            "checker_results": {k: v.model_dump() for k, v in checker_results.items()},
            "corrector_metrics": {
                "tokens_input": existing_metrics["tokens_input"] + total_tokens.prompt_tokens,
                "tokens_output": existing_metrics["tokens_output"] + total_tokens.completion_tokens,
                "total_tokens": existing_metrics["total_tokens"] + total_tokens.total_tokens,
                "estimated_cost_usd": existing_metrics["estimated_cost_usd"] + total_tokens.estimated_cost_usd,
                "llm_calls": existing_metrics["llm_calls"] + (3 if strategy_used == "llm_query_rewriter" else 2),
            },
            "corrector_history": [{
                "cycle": new_total_attempts,
                "decision": "retry_retrieval",
                "strategy": strategy_used,
                "reason": combined_reason,
                "reformulated_query": reformulated,
            }],
        }

    return {
        "route_decision": "approved",
        "correction_attempts": current_attempts,
        "checker_results": {k: v.model_dump() for k, v in checker_results.items()},
    }
