"""
LangGraph Generation Node for VibraDiag.

Orchestrates prompt assembly, LLM invocation via GroqClient, and updates state with llm_response.
"""

from __future__ import annotations

import logging
from typing import Any

from .client import GroqClient
from .prompt_builder import (
    build_initial_signal_query,
    build_messages_payload,
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger(__name__)


def generation_node(state: Any) -> dict[str, Any]:
    """
    LangGraph generation node.

    Reads from state:
    - user_query: str (optional if signal_analysis present)
    - chat_history / messages: list[dict] (optional, prior conversation turns)
    - merged_context: str (optional, from retrieval)
    - retrieval_results: list (optional, fallback if merged_context empty)
    - signal_analysis / signal_processing_result: dict/object (optional, from DSP pipeline)

    Writes to state:
    - llm_response: str
    - active_query: str (the query sent to LLM)
    - errors: list (if any failure occurs)

    Parameters
    ----------
    state : Any (dict, Pydantic BaseModel, or TypedDict)
        The top-level graph state.

    Returns
    -------
    dict[str, Any]
        Partial state update dictionary with key "llm_response".
    """
    if hasattr(state, "model_dump"):
        state_dict = state.model_dump()
    elif isinstance(state, dict):
        state_dict = state
    else:
        state_dict = dict(state)

    signal_data = (
        state_dict.get("signal_analysis")
        or state_dict.get("signal_processing_result")
    )
    has_signal_context = signal_data is not None

    user_query = state_dict.get("user_query", "").strip()

    if not user_query:
        if has_signal_context:
            logger.info("user_query empty, auto-generating initial signal query from DSP output.")
            user_query = build_initial_signal_query(signal_data)
        else:
            logger.warning("generation_node invoked with empty user_query and no signal_data.")
            return {"llm_response": "Sorunuz boş veya okunamadı. Lütfen geçerli bir soru yazın."}

    chat_history = state_dict.get("chat_history") or state_dict.get("messages")
    is_followup = bool(chat_history)

    merged_context = state_dict.get("merged_context", "")
    if not merged_context:
        text_passages = state_dict.get("text_passages", [])
        if text_passages:
            merged_context = "\n\n".join(
                [f"[Kaynak: {p.get('id', 'N/A')}]\n{p.get('text', '')}" for p in text_passages if p.get("text")]
            )
        else:
            retrieval_results = state_dict.get("retrieval_results", [])
            chunks_text = []
            for res in retrieval_results:
                if hasattr(res, "chunks"):
                    chunks = res.chunks
                elif isinstance(res, dict):
                    chunks = res.get("chunks", [])
                else:
                    chunks = []
                for ch in chunks:
                    text = ch.text if hasattr(ch, "text") else ch.get("text", "")
                    if text:
                        chunks_text.append(text)
            if chunks_text:
                merged_context = "\n\n".join(chunks_text)

    system_prompt = build_system_prompt(
        has_signal_context=has_signal_context,
        state=state_dict,
        is_followup=is_followup,
    )
    user_message = build_user_message(query=user_query, context=merged_context)
    messages_payload = build_messages_payload(
        system_prompt=system_prompt,
        user_message=user_message,
        chat_history=chat_history if isinstance(chat_history, list) else None,
    )

    client = GroqClient()
    try:
        response_text = client.generate(messages=messages_payload)
        logger.info("LLM generation completed successfully (%d chars)", len(response_text))
        return {
            "llm_response": response_text,
            "active_query": user_query,
        }
    except Exception as err:
        logger.error("Failed to generate response in generation_node: %s", err, exc_info=True)
        errors = state_dict.get("errors", [])
        if isinstance(errors, list):
            errors = errors + [f"LLM Generation Error: {err}"]
        else:
            errors = [f"LLM Generation Error: {err}"]

        return {
            "llm_response": "Üzgünüm, yanıt oluşturulurken bir teknik hata oluştu. Lütfen tekrar deneyin.",
            "errors": errors,
        }

