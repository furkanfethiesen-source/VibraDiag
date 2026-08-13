"""LangGraph node factory fonksiyonları — query decomposer graph entegrasyonu."""

from __future__ import annotations

from typing import Any

from loguru import logger

from .classifier import QueryComplexityClassifier
from .llm_decomposer import LLMDecomposer


def make_lr_classifier_node(classifier: QueryComplexityClassifier):
    """LR complexity classifier graph node'u üretir."""
    def lr_classifier_node(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        result = classifier.predict(query)
        logger.info(
            f"LR Classifier: is_complex={result.is_complex}, "
            f"confidence={result.confidence:.3f}, query='{query[:80]}'"
        )
        return {
            "is_complex": result.is_complex,
            "classifier_confidence": result.confidence,
        }
    return lr_classifier_node


def make_decomposer_node(decomposer: LLMDecomposer):
    """LLM decomposer graph node'u üretir."""
    def decomposer_node(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        result = decomposer.decompose(query)
        logger.info(
            f"LLM Decomposer: {len(result.sub_queries)} sub-query üretildi, "
            f"reasoning='{(result.reasoning or '')[:100]}'"
        )
        return {
            "sub_queries": result.sub_queries,
            "decomposer_reasoning": result.reasoning,
        }
    return decomposer_node
