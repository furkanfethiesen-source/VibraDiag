"""
LangChain 0.4+ / RAGAS 0.2.15 Compatibility Adapter Module
==========================================================
Ragas 0.2.15 imports `langchain_community.chat_models.vertexai` at top-level in `ragas.llms.base`.
In LangChain 0.3+, VertexAI was relocated to `langchain_google_vertexai`.
This module injects the missing module alias into `sys.modules` before RAGAS is loaded.
"""

from __future__ import annotations

import sys
import types


def apply_ragas_compatibility_patch() -> None:
    """Injects compatibility shim for langchain_community.chat_models.vertexai."""
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        try:
            import langchain_google_vertexai

            v_mod = types.ModuleType("langchain_community.chat_models.vertexai")
            v_mod.ChatVertexAI = langchain_google_vertexai.ChatVertexAI
            sys.modules["langchain_community.chat_models.vertexai"] = v_mod
        except ImportError:
            pass
