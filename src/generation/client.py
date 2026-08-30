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
from langsmith import traceable
from config_loader import load_appcfg

logger = logging.getLogger(__name__)


class GroqClient:
    """Wrapper around Groq API client for generation tasks with Gemini fallback."""

    def __init__(self, config: dict[str, Any] | None = None, api_key: str | None = None):
        app_cfg = load_appcfg()
        llm_cfg = app_cfg.llm if hasattr(app_cfg, "llm") else {}

        provided_cfg = config or llm_cfg
        self.model = provided_cfg.get("model", "openai/gpt-oss-120b")
        self.temperature = provided_cfg.get("temperature", 0.2)
        self.max_tokens = provided_cfg.get("max_tokens", 4096)
        self.reasoning_format = provided_cfg.get("reasoning_format", "parsed")

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

    def _clean_reasoning_tags(self, content: str) -> str:
        """Strip internal chain-of-thought <think> tags cleanly without prompt leakage."""
        if not content:
            return ""

        if "<think>" in content:
            if "</think>" in content:
                parts = content.split("</think>", 1)
                cleaned = parts[1].strip()
            else:
                before_think = content.split("<think>", 1)[0].strip()
                if before_think:
                    cleaned = before_think
                else:
                    cleaned = content.replace("<think>", "").strip()
        else:
            cleaned = content.strip()

        return cleaned if cleaned else content.strip()

    def _generate_with_gemini_fallback(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        """Fallback to Google Gemini if Groq is unavailable or rate-limited."""
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY is not set. Gemini fallback is unavailable.")
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)

            sys_instruction = ""
            user_contents = []
            for msg in messages:
                role = msg.get("role")
                text = msg.get("content", "")
                if role == "system":
                    sys_instruction = (sys_instruction + "\n" + text).strip()
                elif role == "user":
                    user_contents.append(f"Kullanıcı: {text}")
                elif role == "assistant":
                    user_contents.append(f"Asistan: {text}")

            prompt_body = "\n\n".join(user_contents)
            gemini_max = min(max_tokens, 65536)
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=gemini_max,
                system_instruction=sys_instruction if sys_instruction else None,
            )

            logger.info("Calling Gemini Fallback model: gemini-3.1-flash-lite")
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt_body,
                config=config,
            )
            return (response.text or "").strip()
        except Exception as e:
            logger.warning(f"Gemini fallback generation failed: {e}")
            return None

    @traceable(name="groq_generate")
    def generate(
        self,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Executes a chat completion call with Groq and automatic Gemini fallback."""

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
            "Calling Groq LLM model: %s (%d messages, temp=%.2f, max_tokens=%d, reasoning_format=%s)",
            self.model,
            len(messages),
            temp,
            tokens,
            self.reasoning_format,
        )

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    reasoning_format=self.reasoning_format,
                )
                choice = resp.choices[0]
                content = choice.message.content or ""

                # finish_reason kontrolü: kesilme tespiti
                if choice.finish_reason == "length":
                    logger.warning(
                        "⚠️ LLM output truncated (finish_reason='length'). "
                        "Output length: %d chars, max_tokens: %d. "
                        "Consider increasing max_tokens in config.",
                        len(content), tokens,
                    )

                # reasoning_format="parsed" ise content zaten temiz,
                # "raw" ise eski temizleme mantığı çalışır
                if self.reasoning_format == "parsed":
                    return content.strip()
                return self._clean_reasoning_tags(content)

            except RateLimitError as rle:
                last_error = rle
                sleep_time = 2.0 * (2 ** attempt)
                logger.warning(
                    f"Groq RateLimit (429) uyarısı (Deneme {attempt + 1}/{max_retries}): {sleep_time:.1f}sn bekleniyor..."
                )
                time.sleep(sleep_time)

            except Exception as err:
                last_error = err
                logger.warning(f"Groq LLM attempt {attempt + 1} failed: {err}")
                if attempt < max_retries - 1:
                    time.sleep(1.0)

        # Groq exhausted, try Gemini fallback before failing
        logger.warning("Groq retries exhausted, attempting Gemini fallback...")
        fallback_resp = self._generate_with_gemini_fallback(messages, temp, tokens)
        if fallback_resp:
            logger.info("Gemini fallback succeeded (%d chars)", len(fallback_resp))
            return fallback_resp

        logger.error("Groq LLM generation error after all retries: %s", last_error, exc_info=True)
        raise RuntimeError(f"LLM generation failed: {last_error}") from last_error

