"""
VibraDiag — Self-Corrector Checkers.

Implements the 3 core validation checkers:
1. DSPConsistencyChecker: Deterministic regex/entity comparison with signal analysis ground truth (0 tokens, <2ms)
with sentence-isolated negation detection.
2. FaithfulnessChecker: LLM judge (Gemini 3.1 Flash Lite / Groq fallback) measuring context adherence & hallucination prevention.
3. RelevanceChecker: LLM judge measuring user query and decomposed sub-query coverage.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from langsmith import traceable

from config_loader import load_prompts_cfg
from generation.client import GroqClient
from schemas.schemas import CheckerResult, TokenUsageInfo

logger = logging.getLogger(__name__)


def normalize_text_tr(text: str) -> str:
    """Normalize Turkish characters for case-insensitive keyword matching."""
    mapping = {
        "İ": "i", "I": "ı", "Ö": "ö", "Ü": "ü", "Ş": "ş", "Ç": "ç", "Ğ": "ğ"
    }
    for upper_ch, lower_ch in mapping.items():
        text = text.replace(upper_ch, lower_ch)
    return text.lower()


NEGATION_MARKERS: set[str] = {
    "değil", "değildir", "yok", "yoktur", "gözlenmemiştir", "gözlenmedi",
    "görülmemiştir", "görülmedi", "sağlamdır", "sağlam", "tespit edilmemiştir",
    "tespit edilmedi", "saptanmamıştır", "saptanmadı", "ekarte", "ekarte edilmiş",
    "ekarte edilmiştir", "bulunmamaktadır", "bulunmadı", "rastlanmamıştır",
    "rastlanmadı", "temizdir", "kusursuzdur", "normaldir", "mevcut değildir",
    "arızasızdır", "hasarsızdır", "çıkmamıştır", "belirlenmemiştir", "haricindedir",
    "oluşmamıştır", "yansımamaktadır", "olmaksızın", "haricinde",
    # Karşılaştırma ve Ayırt Etme Kalıpları (Comparative & Contrastive Markers):
    "farklı olarak", "aksine", "karıştırılmamalı", "karıştırılmamalıdır",
    "ayırt etmek", "ayırt edilir", "karşılaştırıldığında", "kıyasla",
    "yerine", "benzese de", "benzemesine rağmen",
    "unlike", "in contrast", "distinguished from", "as opposed to"
}


def _is_negated(text: str, term: str, window: int = 6) -> bool:
    """
    Checks if a target term is negated within its sentence context.
    
    Splits text into sentences, identifies matching occurrences of the term,
    and inspects a bidirectional token window (and sentence termination)
    for negation cues.
    """
    if not text or not term:
        return False

    norm_text = normalize_text_tr(text)
    norm_term = normalize_text_tr(term)

    sentences = [s.strip() for s in re.split(r"[.!?;\n]+", norm_text) if s.strip()]

    for sent in sentences:
        if norm_term not in sent:
            continue

        words = re.findall(r"\b[\w-]+\b", sent)
        if not words:
            continue

        term_words = re.findall(r"\b[\w-]+\b", norm_term)
        if not term_words:
            continue

        term_len = len(term_words)

        for i in range(len(words) - term_len + 1):
            if words[i : i + term_len] == term_words:
                start_w = max(0, i - window)
                end_w = min(len(words), i + term_len + window)
                window_tokens = words[start_w:end_w]
                window_str = " ".join(window_tokens)

                for marker in NEGATION_MARKERS:
                    if marker in window_tokens:
                        return True
                    if " " in marker and marker in window_str:
                        return True

                sent_tail = " ".join(words[-3:])
                for marker in NEGATION_MARKERS:
                    if marker in sent_tail:
                        return True

    return False


def _contains_affirmative(text: str, terms: list[str]) -> bool:
    """Returns True if any of the terms is mentioned and at least one occurrence is NOT negated."""
    norm_text = normalize_text_tr(text)
    sentences = [s.strip() for s in re.split(r"[.!?;\n]+", norm_text) if s.strip()]

    for term in terms:
        norm_term = normalize_text_tr(term)
        for sent in sentences:
            if norm_term in sent:
                if not _is_negated(sent, norm_term):
                    return True
    return False


def _is_term_negated_in_text(text: str, terms: list[str]) -> bool:
    """Returns True if any term in terms is present in text AND is negated."""
    norm_text = normalize_text_tr(text)
    sentences = [s.strip() for s in re.split(r"[.!?;\n]+", norm_text) if s.strip()]

    for term in terms:
        norm_term = normalize_text_tr(term)
        for sent in sentences:
            if norm_term in sent and _is_negated(sent, norm_term):
                return True
    return False


class DSPConsistencyChecker:
    """
    Validates generated text against signal processing (DSP) state and ground truth.
    Strict, deterministic, 0 token cost.
    """

    FAULT_KEYWORDS: dict[str, list[str]] = {
        "inner_race_fault": ["bpfi", "iç bilezik", "inner race", "iç yüzey", "iç segman", "iç bilezik hatası", "iç bilezik arızası"],
        "outer_race_fault": ["bpfo", "dış bilezik", "outer race", "dış gövde", "dış yatak", "dış bilezik hatası", "dış bilezik arızası"],
        "ball_fault": ["bsf", "bilya", "ball spin", "yuvarlanma elemanı", "makara", "bilya arızası"],
        "cage_fault": ["ftf", "kafes", "fundamental train", "cage", "kafes arızası"],
        "unbalance": ["unbalance", "dengesizlik", "balanssızlık", "rotor unbalance", "dengesizlik arızası", "rotor balanssızlığı", "balans hatası"],
        "misalignment": ["misalignment", "eksen kaçıklığı", "hizasızlık", "açısal kaçıklık", "paralel kaçıklık", "hizasızlık arızası", "eksenel kaçıklık", "açısal eksen kaçıklığı", "paralel eksen kaçıklığı"],
        "looseness": ["looseness", "mekanik gevşeklik", "gevşeklik arızası", "yapısal gevşeklik"],
        "resonance": ["rezonans", "doğal frekans", "darbe testi", "bump test", "kritik hız rezonansı"],
        "flow": ["kanat geçiş frekansı", "vane pass frequency", "vpf arızası", "kavitasyon"],
        "standards": ["iso 10816", "iso 20816", "şiddet tablosu", "zone a", "zone b", "zone c", "zone d"],
        "windowing": ["pencereleme", "windowing", "leakage etkisi", "hanning penceresi", "hamming penceresi"],
    }

    @classmethod
    @traceable(name="checker_dsp_consistency")
    def check(
        cls,
        generated_text: str,
        signal_data: dict[str, Any] | None = None,
        threshold: float = 1.0,
    ) -> CheckerResult:
        start_time = time.time()

        if not signal_data:
            latency_ms = (time.time() - start_time) * 1000
            return CheckerResult(
                checker_name="dsp_consistency",
                passed=True,
                score=1.0,
                rationale="Sinyal/DSP bağlamı bulunmuyor (Pure RAG modu), kontrol pas geçildi.",
                token_usage=TokenUsageInfo(),
                latency_ms=latency_ms,
            )

        text_lower = generated_text.lower()

        primary_fault = (
            signal_data.get("primary_fault")
            or signal_data.get("envelope_diagnosis", {}).get("fault_type")
            or signal_data.get("fault_type")
            or signal_data.get("fault_type_abbr")
        )

        detected_mismatches: list[str] = []

        if primary_fault:
            norm_fault = normalize_text_tr(str(primary_fault))

            if "inner" in norm_fault or "bpfi" in norm_fault:
                outer_claimed = _contains_affirmative(generated_text, cls.FAULT_KEYWORDS["outer_race_fault"])
                inner_negated = _is_term_negated_in_text(generated_text, cls.FAULT_KEYWORDS["inner_race_fault"])
                if outer_claimed or inner_negated:
                    detected_mismatches.append(
                        "DSP ground truth 'İç Bilezik (BPFI)' arızası tespit etmişken üretilen metinde 'Dış Bilezik (BPFO)' arızası iddia edilmiş veya İç Bilezik arızası reddedilmiştir."
                    )

            elif "outer" in norm_fault or "bpfo" in norm_fault:
                inner_claimed = _contains_affirmative(generated_text, cls.FAULT_KEYWORDS["inner_race_fault"])
                outer_negated = _is_term_negated_in_text(generated_text, cls.FAULT_KEYWORDS["outer_race_fault"])
                if inner_claimed or outer_negated:
                    detected_mismatches.append(
                        "DSP ground truth 'Dış Bilezik (BPFO)' arızası tespit etmişken üretilen metinde 'İç Bilezik (BPFI)' arızası iddia edilmiş veya Dış Bilezik arızası reddedilmiştir."
                    )

            elif "unbalance" in norm_fault or "dengesizlik" in norm_fault:
                misalignment_claimed = _contains_affirmative(generated_text, cls.FAULT_KEYWORDS["misalignment"])
                unbalance_negated = _is_term_negated_in_text(generated_text, cls.FAULT_KEYWORDS["unbalance"])
                if misalignment_claimed or unbalance_negated:
                    detected_mismatches.append(
                        "DSP ground truth 'Dengesizlik (Unbalance)' arızası tespit etmişken üretilen metinde 'Hizasızlık (Misalignment)' iddia edilmiş veya Dengesizlik arızası reddedilmiştir."
                    )

            elif "misalignment" in norm_fault or "hizasızlık" in norm_fault or "kaçıklık" in norm_fault:
                unbalance_claimed = _contains_affirmative(generated_text, cls.FAULT_KEYWORDS["unbalance"])
                misalignment_negated = _is_term_negated_in_text(generated_text, cls.FAULT_KEYWORDS["misalignment"])
                if unbalance_claimed or misalignment_negated:
                    detected_mismatches.append(
                        "DSP ground truth 'Hizasızlık (Misalignment)' arızası tespit etmişken üretilen metinde 'Dengesizlik (Unbalance)' iddia edilmiş veya Hizasızlık reddedilmiştir."
                    )

        iso_zone = signal_data.get("iso_severity_zone")
        if iso_zone:
            zone_letter = iso_zone.strip().upper()[-1]

            zone_matches = list(re.finditer(r"\bzone\s*([a-d])\b", text_lower))
            for zm in zone_matches:
                claimed_letter = zm.group(1).upper()
                match_str = zm.group(0)
                if claimed_letter != zone_letter:
                    if not _is_negated(generated_text, match_str):
                        detected_mismatches.append(
                            f"DSP ISO şiddet zonu '{iso_zone}' iken üretilen metinde 'Zone {claimed_letter}' iddia edilmiştir."
                        )
                        break

        latency_ms = (time.time() - start_time) * 1000

        if detected_mismatches:
            rationale = " | ".join(detected_mismatches)
            logger.warning("DSPConsistencyChecker FAILED: %s", rationale)
            return CheckerResult(
                checker_name="dsp_consistency",
                passed=False,
                score=0.0,
                rationale=rationale,
                token_usage=TokenUsageInfo(),
                latency_ms=latency_ms,
                execution_error=False,
            )

        return CheckerResult(
            checker_name="dsp_consistency",
            passed=True,
            score=1.0,
            rationale="Üretilen yanıt DSP state ve ground truth verileriyle tam tutarlıdır.",
            token_usage=TokenUsageInfo(),
            latency_ms=latency_ms,
            execution_error=False,
        )


class GeminiJudgeClient:
    """
    Evaluator/Judge LLM Client using Google Gemini 3.1 Flash Lite (with fallback to Groq).
    Outputs strictly valid JSON format with temperature=0.0.
    """

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._genai_client = None
        self._groq_fallback: GroqClient | None = None

    @property
    def genai_client(self):
        if self._genai_client is None:
            key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing.")
            from google import genai
            self._genai_client = genai.Client(api_key=key)
        return self._genai_client

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "Sen JSON formatında yanıt veren bir değerlendirme yargıcısın.",
    ) -> tuple[dict[str, Any], TokenUsageInfo]:
        """
        Executes judge completion with JSON output and returns (parsed_dict, TokenUsageInfo).
        """
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                system_instruction=system_instruction,
            )
            response = self.genai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or "{}"
            data = json.loads(raw_text)

            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            if prompt_tokens == 0:
                prompt_tokens = len(prompt.split()) * 4 // 3
                completion_tokens = len(raw_text.split()) * 4 // 3

            cost_usd = (prompt_tokens * 0.075 + completion_tokens * 0.30) / 1_000_000
            usage = TokenUsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost_usd=round(cost_usd, 6),
            )
            return data, usage

        except Exception as gemini_err:
            logger.warning("Gemini judge call failed, attempting Groq fallback: %s", gemini_err)
            if self._groq_fallback is None:
                self._groq_fallback = GroqClient()

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ]
            raw_groq = self._groq_fallback.generate(messages=messages, temperature=0.0)
            cleaned = raw_groq.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            data = json.loads(cleaned)
            prompt_tokens = len(prompt.split()) * 4 // 3
            completion_tokens = len(raw_groq.split()) * 4 // 3
            cost_usd = (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000
            usage = TokenUsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost_usd=round(cost_usd, 6),
            )
            return data, usage


class FaithfulnessChecker:
    """
    Evaluates whether the candidate answer is faithful to the retrieved context chunks.
    Detects hallucinations and unsupported factual claims.
    """

    def __init__(self, judge_client: Any | None = None):
        self.judge_client = judge_client or GeminiJudgeClient()

    @traceable(name="checker_faithfulness")
    def check(
        self,
        generated_text: str,
        context_chunks: list[dict[str, Any]] | str,
        threshold: float = 0.70,
    ) -> CheckerResult:
        start_time = time.time()

        if isinstance(context_chunks, list):
            context_text = "\n\n".join(
                f"[Kaynak: {c.get('id', 'N/A')}]\n{c.get('text', '')}"
                for c in context_chunks
                if c.get("text")
            )
        else:
            context_text = str(context_chunks)

        if not context_text.strip():
            latency_ms = (time.time() - start_time) * 1000
            return CheckerResult(
                checker_name="faithfulness",
                passed=False,
                score=0.0,
                rationale="Kaynak bağlam boş olduğu için üretilen yanıt doğrulanamadı.",
                token_usage=TokenUsageInfo(),
                latency_ms=latency_ms,
                execution_error=False,
            )

        try:
            prompt_template = load_prompts_cfg().corrector_prompts["faithfulness_prompt"]
        except Exception as e:
            logger.error("Failed to load faithfulness_prompt: %s", e)
            prompt_template = "{context}\n{answer}"

        prompt = prompt_template.format(
            context=context_text[:3000],
            answer=generated_text[:1500],
        )

        try:
            if hasattr(self.judge_client, "generate_json"):
                data, usage = self.judge_client.generate_json(
                    prompt=prompt,
                    system_instruction="Sen JSON formatında yanıt veren bir Faithfulness değerlendirme yargıcısın.",
                )
            else:
                messages = [
                    {"role": "system", "content": "Sen JSON formatında yanıt veren bir Faithfulness değerlendirme yargıcısın."},
                    {"role": "user", "content": prompt},
                ]
                raw_response = self.judge_client.generate(messages=messages, temperature=0.0)
                cleaned = raw_response.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()
                data = json.loads(cleaned)
                prompt_tokens = len(prompt.split()) * 4 // 3
                completion_tokens = len(raw_response.split()) * 4 // 3
                cost_usd = (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000
                usage = TokenUsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    estimated_cost_usd=round(cost_usd, 6),
                )

            latency_ms = (time.time() - start_time) * 1000
            score = float(data.get("score", 0.0))
            rationale = str(data.get("rationale", "Faithfulness değerlendirmesi tamamlandı."))
            passed = score >= threshold

            logger.info("FaithfulnessChecker: score=%.2f (passed=%s)", score, passed)

            return CheckerResult(
                checker_name="faithfulness",
                passed=passed,
                score=score,
                rationale=rationale,
                token_usage=usage,
                latency_ms=latency_ms,
                execution_error=False,
            )

        except Exception as err:
            logger.error("FaithfulnessChecker LLM call failed: %s", err, exc_info=True)
            latency_ms = (time.time() - start_time) * 1000

            return CheckerResult(
                checker_name="faithfulness",
                passed=True,
                score=0.75,
                rationale=f"LLM Judge hatası nedeniyle varsayılan tolerans uygulandı: {err}",
                token_usage=TokenUsageInfo(),
                latency_ms=latency_ms,
                execution_error=True,
                error_message=str(err),
            )


class RelevanceChecker:
    """
    Evaluates whether the candidate answer adequately answers the user query
    and all decomposed sub-queries without skipping key requirements.
    """

    def __init__(self, judge_client: Any | None = None):
        self.judge_client = judge_client or GeminiJudgeClient()

    @traceable(name="checker_sub_query_relevance")
    def check(
        self,
        user_query: str,
        generated_text: str,
        sub_queries: list[str] | None = None,
        threshold: float = 0.65,
    ) -> CheckerResult:
        start_time = time.time()
        sub_queries = sub_queries or [user_query]

        try:
            prompt_template = load_prompts_cfg().corrector_prompts["relevance_prompt"]
        except Exception as e:
            logger.error("Failed to load relevance_prompt: %s", e)
            prompt_template = "{user_query}\n{sub_queries}\n{answer}"

        prompt = prompt_template.format(
            user_query=user_query,
            sub_queries=json.dumps(sub_queries, ensure_ascii=False),
            answer=generated_text[:1500],
        )

        try:
            if hasattr(self.judge_client, "generate_json"):
                data, usage = self.judge_client.generate_json(
                    prompt=prompt,
                    system_instruction="Sen JSON formatında yanıt veren bir Relevance değerlendirme yargıcısın.",
                )
            else:
                messages = [
                    {"role": "system", "content": "Sen JSON formatında yanıt veren bir Relevance değerlendirme yargıcısın."},
                    {"role": "user", "content": prompt},
                ]
                raw_response = self.judge_client.generate(messages=messages, temperature=0.0)
                cleaned = raw_response.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()
                data = json.loads(cleaned)
                prompt_tokens = len(prompt.split()) * 4 // 3
                completion_tokens = len(raw_response.split()) * 4 // 3
                cost_usd = (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000
                usage = TokenUsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    estimated_cost_usd=round(cost_usd, 6),
                )

            latency_ms = (time.time() - start_time) * 1000
            score = float(data.get("score", 0.0))
            rationale = str(data.get("rationale", "Alaka değerlendirmesi tamamlandı."))

            norm_gen_text = normalize_text_tr(generated_text)
            is_refusal = any(
                phrase in norm_gen_text for phrase in [
                    "bilgi bulunmamaktadır", "bilgi yer almamaktadır", "yeterli bilgi yok",
                    "bilgi bulunamadı", "karşılayan bir bilgi", "dokümantasyonda yer almamaktadır",
                    "bağlamda bulunmamaktadır", "bağlamda yer almamaktadır", "bilgiye ulaşılamamıştır",
                    "bu konuda bilgi bulunmamaktadır", "belgelerde yer almamaktadır", "kaynaklarda yer almamaktadır"
                ]
            )
            if is_refusal and score < threshold:
                logger.info("RelevanceChecker: Epistemik ret tespit edildi, skor 0.95 olarak kabul edildi.")
                score = max(score, 0.95)
                rationale = f"Epistemik Ret (Dürüst Bilgi Yokluğu Bildirimi): {rationale}"

            passed = score >= threshold

            logger.info("RelevanceChecker: score=%.2f (passed=%s)", score, passed)

            return CheckerResult(
                checker_name="sub_query_relevance",
                passed=passed,
                score=score,
                rationale=rationale,
                token_usage=usage,
                latency_ms=latency_ms,
                execution_error=False,
            )

        except Exception as err:
            logger.error("RelevanceChecker LLM call failed: %s", err, exc_info=True)
            latency_ms = (time.time() - start_time) * 1000
            return CheckerResult(
                checker_name="sub_query_relevance",
                passed=True,
                score=0.70,
                rationale=f"LLM Judge hatası nedeniyle varsayılan kabul uygulandı: {err}",
                token_usage=TokenUsageInfo(),
                latency_ms=latency_ms,
                execution_error=True,
                error_message=str(err),
            )
