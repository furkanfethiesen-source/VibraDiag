
"""
VibraDiag Retrieval/Reranker Evaluation Metrics File
====================================================
Deterministik ID-seviyesi metrikler (LLM Judge gerektirmez):
- recall_at_k
- precision_at_k
- hit_rate_at_k
- mrr_at_k
- ndcg_at_k
"""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = None) -> float:
    """
    Retrieved ID'lerin Top-K içindeki Recall değerini hesaplar.
    """
    if not expected_ids:
        return 0.0
    cutoff = retrieved_ids[:k] if k is not None else retrieved_ids
    if not cutoff:
        return 0.0

    retrieved_set = set(cutoff)
    expected_set = set(expected_ids)
    hits = len(retrieved_set & expected_set)
    return hits / len(expected_set)


def precision_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = None) -> float:
    """
    Retrieved ID'lerin Top-K içindeki Precision değerini hesaplar.
    """
    cutoff = retrieved_ids[:k] if k is not None else retrieved_ids
    if not cutoff or not expected_ids:
        return 0.0

    retrieved_set = set(cutoff)
    expected_set = set(expected_ids)
    hits = len(retrieved_set & expected_set)
    return hits / len(cutoff)


def hit_rate_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = None) -> float:
    """
    Retrieved ID'lerin Top-K içinde en az 1 tane beklenen ID ile eşleşip eşleşmediğini ölçer.
    """
    cutoff = retrieved_ids[:k] if k is not None else retrieved_ids
    if not cutoff or not expected_ids:
        return 0.0

    retrieved_set = set(cutoff)
    expected_set = set(expected_ids)
    return 1.0 if bool(retrieved_set & expected_set) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = None) -> float:
    """
    Mean Reciprocal Rank @ K hesaplar. İlk eşleşen dokümanın sırasına göre 1/rank döner.
    """
    cutoff = retrieved_ids[:k] if k is not None else retrieved_ids
    if not cutoff or not expected_ids:
        return 0.0

    expected_set = set(expected_ids)
    for rank, doc_id in enumerate(cutoff, start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = None) -> float:
    """
    Normalized Discounted Cumulative Gain @ K (Binary relevance) hesaplar.
    """
    cutoff = retrieved_ids[:k] if k is not None else retrieved_ids
    if not cutoff or not expected_ids:
        return 0.0

    expected_set = set(expected_ids)
    dcg = 0.0
    for rank, doc_id in enumerate(cutoff, start=1):
        if doc_id in expected_set:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected_set), len(cutoff)) + 1))
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def extract_parent_id(chunk_id: str, metadata: dict | None = None) -> str:
    """
    Bir child chunk ID'sinden veya metadata'sından parent ID türetir.
    Ör: 'text_parent_15_child_0' -> 'text_parent_15'
    """
    if metadata:
        p_id = metadata.get("parent_id") or metadata.get("parent_doc_id")
        if p_id:
            return str(p_id)
    if "_child_" in chunk_id:
        return chunk_id.split("_child_")[0]
    return chunk_id





                



