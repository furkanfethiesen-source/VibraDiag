"""
LLM Client Wrapper for VibraDiag Generation Layer.

Provides a unified interface to execute completion calls against Groq (or fallback providers).
"""

from __future__ import annotations

import logging
from typing import Any

from groq import Groq

from config_loader import load_appcfg

logger = logging.getLogger(__name__)


class GroqClient:
    """Wrapper around Groq API client for generation tasks."""

    def __init__(self, config: dict[str, Any] | None = None, api_key: str | None = None):
        app_cfg = load_appcfg()
        llm_cfg = app_cfg.llm if hasattr(app_cfg, "llm") else {}

        provided_cfg = config or llm_cfg
        self.model = provided_cfg.get("model", "llama-3.3-70b-versatile")
        self.temperature = provided_cfg.get("temperature", 0.3)
        self.max_tokens = provided_cfg.get("max_tokens", 2048)

        if api_key:
            self._client = Groq(api_key=api_key)
        else:
            self._client = Groq()

    def generate(
        self,
        system_prompt: str | None = None,
        user_message: str | None = None,
        messages: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate LLM response given system prompt & user message OR a list of messages.

        Parameters
        ----------
        system_prompt : str, optional
            The system prompt containing guidelines and optional signal context.
        user_message : str, optional
            The user query and/or retrieved documentation context.
        messages : list of dict, optional
            Full chat history payload [{"role": "...", "content": "..."}, ...]
        temperature : float, optional
            Override default temperature.
        max_tokens : int, optional
            Override default max tokens.

        Returns
        -------
        str
            Generated text content.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if messages is None:
            sys_p = system_prompt or ""
            usr_m = user_message or ""
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_m},
            ]

        logger.info(
            "Calling Groq LLM model: %s (%d messages, temp=%.2f, max_tokens=%d)",
            self.model,
            len(messages),
            temp,
            tokens,
        )

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            content = resp.choices[0].message.content or ""
            return content
        except Exception as err:
            logger.error("Groq LLM generation error: %s", err, exc_info=True)
            raise RuntimeError(f"LLM generation failed: {err}") from err

