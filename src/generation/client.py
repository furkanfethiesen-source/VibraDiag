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
from .prompt_builder import repair_truncated_markdown

logger = logging.getLogger(__name__)


class GroqClient:
    """Wrapper around Groq API client for generation tasks with Gemini fallback."""

    def __init__(self, config: dict[str, Any] | None = None, api_key: str | None = None):
        app_cfg = load_appcfg()
        llm_cfg = dict(app_cfg.llm) if hasattr(app_cfg, "llm") and app_cfg.llm else {}

        provided_cfg = {**llm_cfg, **(config or {})}
        self.model = provided_cfg.get("model", "openai/gpt-oss-120b")
        self.temperature = float(provided_cfg.get("temperature", 0.2))
        self.max_tokens = int(provided_cfg.get("max_tokens", 2400))
        self.reasoning_format = provided_cfg.get("reasoning_format", "parsed")

        self.last_finish_reason: str = "stop"
        self.was_truncated: bool = False

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

    def _stitch_continuation(self, partial: str, continuation: str) -> str:
        """
        Kısmi metin ile devam metnini tekrarları temizleyerek dikişsiz birleştirir.
        """
        if not continuation:
            return partial
        if not partial:
            return continuation

        p_stripped = partial.rstrip()
        c_stripped = continuation.lstrip()

        # Eğer continuation, partial'ın son birkaç kelimesiyle başlıyorsa tekrarı önle
        words_p = p_stripped.split()
        for overlap_len in range(min(len(words_p), 8), 0, -1):
            overlap_phrase = " ".join(words_p[-overlap_len:]).lower()
            if c_stripped.lower().startswith(overlap_phrase):
                c_stripped = c_stripped[len(overlap_phrase):].lstrip()
                break

        sep = " "
        if p_stripped.endswith(("\n", " ", "-", "—")):
            sep = ""

        stitched = p_stripped + sep + c_stripped
        return repair_truncated_markdown(stitched)

    def _continue_truncated_response(
        self,
        messages: list[dict[str, str]],
        partial_content: str,
        temperature: float,
    ) -> str:
        """
        finish_reason == 'length' durumunda modelin baştan başlamasını engelleyerek
        kaldığı yerden tamamlamasını sağlayan tek seferlik ek çağrı yapar.
        """
        if not partial_content or len(partial_content.strip()) < 30:
            return partial_content

        clean_partial = (
            partial_content.strip()
            if self.reasoning_format == "parsed"
            else self._clean_reasoning_tags(partial_content).rstrip()
        )
        last_snippet = clean_partial[-100:].replace("\n", " ")

        continuation_messages = list(messages)
        continuation_messages.append({"role": "assistant", "content": clean_partial})

        follow_up_prompt = (
            f"[SİSTEM TAMAMLAMA DİREKTİFİ]\n"
            f"Önceki yanıtın uzunluk sınırına ulaştığı için tam olarak şu noktada yarıda kesildi:\n"
            f"\"{last_snippet}\"\n\n"
            f"KESİN KURALLAR:\n"
            f"1. ASLA baştan başlama, giriş cümlesi kurma, özetleme veya önceki yazdıklarını tekrarlama.\n"
            f"2. YALNIZCA cümlenin/kelimenin kaldığı noktadan itibaren eksik kalan kısımları tamamla.\n"
            f"3. Kalan maddeleri tamamlayıp yanıtı sonlandır."
        )
        continuation_messages.append({"role": "user", "content": follow_up_prompt})

        logger.info(
            "Requesting seamless continuation for truncated response (partial length: %d chars, last: '...%s')",
            len(clean_partial),
            last_snippet[-40:],
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=continuation_messages,
                temperature=temperature,
                max_tokens=800,
                reasoning_format=self.reasoning_format,
            )
            continuation_text = resp.choices[0].message.content or ""
            if self.reasoning_format != "parsed":
                continuation_text = self._clean_reasoning_tags(continuation_text)

            stitched = self._stitch_continuation(clean_partial, continuation_text)
            logger.info(
                "Continuation succeeded (stitched total length: %d chars)", len(stitched)
            )
            return stitched
        except Exception as cont_err:
            logger.warning(
                "Continuation call failed: %s. Returning repaired partial response.",
                cont_err,
            )
            return repair_truncated_markdown(clean_partial)

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

                # finish_reason kontrolü: kesilme tespiti ve dikişsiz tamamlama
                if choice.finish_reason == "length":
                    self.last_finish_reason = "length"
                    self.was_truncated = True
                    logger.warning(
                        "⚠️ LLM output truncated (finish_reason='length'). "
                        "Output length: %d chars, max_tokens: %d. "
                        "Triggering seamless suffix continuation...",
                        len(content), tokens,
                    )
                    content = self._continue_truncated_response(messages, content, temp)
                else:
                    self.last_finish_reason = choice.finish_reason or "stop"
                    self.was_truncated = False

                if self.reasoning_format == "parsed":
                    return repair_truncated_markdown(content.strip())
                return repair_truncated_markdown(self._clean_reasoning_tags(content))

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

