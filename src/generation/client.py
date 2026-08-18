"""
LLM Client Wrapper for VibraDiag Generation Layer.

Provides a unified interface to execute completion calls against Groq (or fallback providers).
"""

from __future__ import annotations

import logging
import logging
import os
from typing import Any
import re
import time
from groq import RateLimitError

from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from config_loader import load_appcfg

logger = logging.getLogger(__name__)


class GroqClient:
    """Wrapper around Groq API client for generation tasks."""

    def __init__(self, config: dict[str, Any] | None = None, api_key: str | None = None):
        app_cfg = load_appcfg()
        llm_cfg = app_cfg.llm if hasattr(app_cfg, "llm") else {}

        provided_cfg = config or llm_cfg
        self.model = provided_cfg.get("model", "qwen/qwen3.6-27b")
        self.temperature = provided_cfg.get("temperature", 0.3)
        self.max_tokens = provided_cfg.get("max_tokens", 2048)

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client = None
        if self.api_key:
            try:
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.debug("Groq client init exception: %s", e)

    @property
    def client(self) -> Groq:
        if self._client is None:
            key = self.api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("GROQ_API_KEY environment variable is missing or empty.")
            self._client = Groq(api_key=key)
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Executes a chat completion call with Groq."""

        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens

        if messages is None:
            sys_p = system_prompt or "You are a helpful assistant."
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                content = resp.choices[0].message.content or ""
                cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return cleaned or content
            except RateLimitError as rle:
                logger.warning(f"Groq RateLimit (429) uyarısı (Deneme {attempt + 1}/{max_retries}): 10sn bekleniyor...")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM generation failed after {max_retries} retries: {rle}") from rle
                time.sleep(10)
            except Exception as err:
                logger.error("Groq LLM generation error: %s", err, exc_info=True)
                raise RuntimeError(f"LLM generation failed: {err}") from err

