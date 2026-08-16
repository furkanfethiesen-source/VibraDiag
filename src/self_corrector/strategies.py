"""
VibraDiag — Self-Corrector Query Reformulation & Retry Strategies.

Implements the 3-tier waterfall retrieval retry strategy:
1. Static Domain Dictionary Query Expansion (Rule-based, 0 tokens, 0ms)
2. Pseudo-Relevance Feedback / PRF (Deterministic, Threshold-Guard protected)
3. Lightweight LLM Query Rewriter (Groq Llama-3.3-70b with failure feedback)
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from langsmith import traceable

from config_loader import load_prompts_cfg
from generation.client import GroqClient
from schemas.schemas import TokenUsageInfo

logger = logging.getLogger(__name__)


def normalize_text_tr(text: str) -> str:
    """Normalize Turkish characters for case-insensitive keyword matching."""
    mapping = {
        "İ": "i", "I": "ı", "Ö": "ö", "Ü": "ü", "Ş": "ş", "Ç": "ç", "Ğ": "ğ"
    }
    for upper_ch, lower_ch in mapping.items():
        text = text.replace(upper_ch, lower_ch)
    return text.lower()


DOMAIN_SYNONYM_MAP: dict[str, dict[str, list[str]]] = {
    "bpfi": {
        "triggers": ["iç bilezik", "bpfi", "inner race", "iç yüzey", "iç segman", "iç bilezik hatası", "iç bilezik arızası"],
        "expansions": ["BPFI", "iç bilezik hatası", "inner race fault", "bilya geçiş frekansı iç"],
    },
    "bpfo": {
        "triggers": ["dış bilezik", "bpfo", "outer race", "dış gövde", "dış yatak", "dış bilezik hatası", "dış bilezik arızası"],
        "expansions": ["BPFO", "dış bilezik hatası", "outer race fault", "bilya geçiş frekansı dış"],
    },
    "bsf": {
        "triggers": ["bilya", "yuvarlanma elemanı", "bsf", "ball spin", "makara", "bilya arızası", "bilya hasarı"],
        "expansions": ["BSF", "bilya arızası", "ball spin frequency", "yuvarlanma elemanı hasarı"],
    },
    "ftf": {
        "triggers": ["kafes", "ftf", "cage", "fundamental train", "kafes arızası", "kafes hasarı"],
        "expansions": ["FTF", "kafes arızası", "fundamental train frequency"],
    },
    "unbalance": {
        "triggers": ["dengesizlik", "balans", "balanssızlık", "unbalance", "1x", "1x rpm", "rotor balans", "rotor unbalance", "dengesizlik arızası"],
        "expansions": ["1X RPM", "balanssızlık", "rotor unbalance", "dengesizlik arızası"],
    },
    "misalignment": {
        "triggers": ["hizasızlık", "eksen kaçıklığı", "misalignment", "2x", "2x rpm", "açısal", "açısal kaçıklık", "paralel", "paralel kaçıklık"],
        "expansions": ["2X RPM", "eksen kaçıklığı", "angular misalignment", "paralel kaçıklık"],
    },
    "looseness": {
        "triggers": ["gevşeklik", "looseness", "harmonik", "harmonikler", "0.5x", "boşluk", "mekanik gevşeklik", "alt harmonik"],
        "expansions": ["mekanik gevşeklik", "0.5X alt harmonik", "structural looseness", "harmonikler"],
    },
    "resonance": {
        "triggers": ["rezonans", "doğal frekans", "darbe testi", "bump test", "kritik hız", "bode"],
        "expansions": ["rezonans", "doğal frekans", "darbe testi", "bump test"],
    },
    "flow": {
        "triggers": ["kanat geçiş", "vane pass", "vpf", "kavitasyon", "akış"],
        "expansions": ["kanat geçiş frekansı", "VPF", "kavitasyon"],
    },
    "standards": {
        "triggers": ["iso 10816", "iso 20816", "şiddet tablosu", "severity", "rms hız", "zone a", "zone b", "zone c", "zone d"],
        "expansions": ["ISO 10816", "titreşim şiddet tablosu", "RMS hız mm/s", "kabul edilebilir sınır"],
    },
    "windowing": {
        "triggers": ["pencere", "windowing", "leakage", "hanning", "hamming", "fft"],
        "expansions": ["pencereleme", "leakage effect", "Hanning penceresi"],
    },
}


class DomainDictionaryExpansion:
    """
    Tier 1: Fast, deterministic domain query expander.
    Expands queries with curated industrial vibration terminology without LLM.
    """

    @classmethod
    def expand(cls, query: str) -> str:
        q_norm = normalize_text_tr(query)
        added_terms: list[str] = []

        for category, cluster in DOMAIN_SYNONYM_MAP.items():
            triggers = cluster["triggers"]
            expansions = cluster["expansions"]

            if any(normalize_text_tr(t) in q_norm for t in triggers):
                for exp in expansions:
                    exp_norm = normalize_text_tr(exp)
                    if exp_norm not in q_norm and exp not in added_terms:
                        added_terms.append(exp)
                        if len(added_terms) >= 3:
                            break

        if added_terms:
            expanded = f"{query} ({', '.join(added_terms[:3])})"
            logger.info("Domain dictionary expanded query: '%s' -> '%s'", query, expanded)
            return expanded
        return query


class PseudoRelevanceFeedback:
    """
    Tier 2: Extract prominent technical terms from top-k retrieved chunks
    only if max chunk score >= min_score_guard (to prevent noise propagation).
    """

    STOP_WORDS = {
        "ve", "ile", "için", "olan", "bir", "bu", "şu", "gibi", "kadar", "daha",
        "çok", "en", "the", "and", "is", "of", "in", "to", "for", "with", "that",
        "are", "as", "by", "on", "at", "from", "an", "be", "this", "which"
    }

    @classmethod
    def extract_feedback_terms(
        cls,
        chunks: list[dict[str, Any]],
        max_score: float,
        min_score_guard: float = 0.40,
        top_n_terms: int = 3,
    ) -> list[str]:
        if max_score < min_score_guard or not chunks:
            logger.info(
                "PRF skipped: max_score (%.3f) below threshold guard (%.3f)",
                max_score,
                min_score_guard,
            )
            return []

        term_freq: dict[str, int] = {}
        for ch in chunks[:3]:
            text = ch.get("text", "")

            words = re.findall(r"\b[A-Za-z0-9İıŞşĞğÜüÇçÖö\-_]{3,}\b", text)
            for w in words:
                w_clean = w.strip()
                if w_clean.lower() not in cls.STOP_WORDS and not w_clean.isdigit():
                    term_freq[w_clean] = term_freq.get(w_clean, 0) + 1

        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        extracted = [term for term, _ in sorted_terms[:top_n_terms]]
        logger.info("PRF extracted salient terms: %s", extracted)
        return extracted

    @classmethod
    def expand_with_prf(
        cls,
        query: str,
        chunks: list[dict[str, Any]],
        max_score: float,
        min_score_guard: float = 0.40,
    ) -> str:
        terms = cls.extract_feedback_terms(chunks, max_score, min_score_guard)
        if terms:
            return f"{query} {' '.join(terms)}"
        return query


class QueryRewriter:
    """
    Tier 3: Rephrase query targeting specific failure reasons using Groq LLM.
    """

    def __init__(self, groq_client: GroqClient | None = None):
        self.groq_client = groq_client or GroqClient()

    @traceable(name="strategy_llm_query_rewrite")
    def rewrite(
        self,
        query: str,
        failure_reason: str = "Bağlam yetersiz veya alakasız chunk döndü.",
    ) -> tuple[str, TokenUsageInfo, float]:
        start_time = time.time()
        
        try:
            prompt_template = load_prompts_cfg().corrector_prompts["rewrite_prompt"]
        except Exception as e:
            logger.error("Failed to load rewrite_prompt: %s", e)
            prompt_template = "{query}\n{failure_reason}"

        prompt = prompt_template.format(
            query=query,
            failure_reason=failure_reason,
        )

        messages = [
            {"role": "system", "content": "Sen bir arama sorgusu optimizasyon asistanısın."},
            {"role": "user", "content": prompt},
        ]

        try:
            rewritten = self.groq_client.generate(messages=messages, temperature=0.2).strip()

            rewritten = rewritten.strip('"').strip("'")
            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = len(prompt.split()) * 4 // 3
            completion_tokens = len(rewritten.split()) * 4 // 3
            total_tokens = prompt_tokens + completion_tokens

            cost_usd = (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000

            usage = TokenUsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=round(cost_usd, 6),
            )
            logger.info("LLM rewritten query: '%s' -> '%s' (Latency: %.1fms)", query, rewritten, latency_ms)
            return rewritten, usage, latency_ms
        except Exception as err:
            logger.warning("LLM query rewrite failed, falling back to dictionary expansion: %s", err)
            fallback = DomainDictionaryExpansion.expand(query)
            latency_ms = (time.time() - start_time) * 1000
            return fallback, TokenUsageInfo(), latency_ms
