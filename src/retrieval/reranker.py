"""
Cross-encoder tabanlı reranker.

Hybrid search sonrası aday dokümanları query ile birlikte cross-encoder'a
vererek yeniden skorlar. Skorların logit değerleri  threshold ile kıyaslanır
ve tutarlı bir karşılaştırma yapılır.
"""

from __future__ import annotations

import math
from typing import Any
import asyncio

from loguru import logger
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder tabanlı reranker.

    Parametreler
    ----------
    model_name : str
        Kullanılacak cross-encoder modeli.
    device : str
        Hesaplama cihazı ("mps", "cuda", "cpu").
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "mps",
        batch_size: int = 32,
        score_threshold: float = 0.35,
    ):
        logger.info(f"Reranker yükleniyor: {model_name} ({device})")
        self.model = CrossEncoder(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size
        self.score_threshold = score_threshold



    def rerank(
        self,
        query: str,
        candidates: dict[str, Any],
        top_k: int = 2,
        score_threshold: float | None = None,
        deduplicate_parents: bool = True,
    ) -> dict[str, Any]:
        """
        Candidate dokümanları cross-encoder ile yeniden skorlar ve threshold altındakileri süzer.

        Parameters
        ----------
        query : str
            Kullanıcı sorgusu.
        candidates : dict
            Retrieval results dict: ``{ids, documents, metadatas, distances}``.
        top_k : int
            Döndürülecek en iyi sonuç sayısı.
        score_threshold : float | None
            Skor alt eşik değeri. None ise constructor'daki değer kullanılır.

        Returns
        -------
        dict
            Aynı retrieval results dict formatı. ``distances`` = logits).
        """
        empty: dict[str, Any] = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        ids = candidates.get("ids", [[]])[0]
        documents = candidates.get("documents", [[]])[0]
        metadatas = candidates.get("metadatas", [[]])[0]

        if not ids or not documents:
            return empty

        threshold = score_threshold if score_threshold is not None else self.score_threshold
        pairs = [(query, str(doc)) for doc in documents]
        raw_scores: Any = self.model.predict(pairs, batch_size=self._batch_size)

        scored = []
        for i, raw_score in enumerate(raw_scores):
            # Sigmoid normalizasyonu: Cross-Encoder ham logit değerlerini [0, 1] olasılık alanına çek
            prob = 1.0 / (1.0 + math.exp(-float(raw_score)))
            scored.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": prob,
                    "raw_score": float(raw_score),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)

        filtered = [item for item in scored if item["score"] >= threshold]

        if deduplicate_parents:
            seen_parents: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for item in filtered:
                meta = item.get("metadata") or {}
                parent_id = meta.get("parent_id") or meta.get("parent_doc_id")
                if parent_id:
                    if parent_id in seen_parents:
                        continue
                    seen_parents.add(parent_id)
                deduped.append(item)
                if len(deduped) >= top_k:
                    break
            top = deduped if deduped else filtered[:top_k]
        else:
            top = filtered[:top_k]

        logger.debug(
            f"Rerank tamamlandı: {len(ids)} aday → {len(top)} sonuç (threshold={threshold:.2f}, dedup={deduplicate_parents}) | "
            f"max_score={top[0]['score']:.3f} (raw={top[0]['raw_score']:.2f})" if top else "Rerank: boş sonuç"
        )

        return {
            "ids": [[item["id"] for item in top]],
            "documents": [[item["document"] for item in top]],
            "metadatas": [[item["metadata"] for item in top]],
            "distances": [[item["score"] for item in top]],
            "raw_scores": [[item["raw_score"] for item in top]],
        }

    async def arerank(
        self,
        query: str,
        candidates: dict[str, Any],
        top_k: int = 2,
        score_threshold: float | None = None,
        deduplicate_parents: bool = True,
    ) -> dict[str, Any]:
        """Rerank'ın asenkron wrapper'ı — CPU-bound işlemi thread pool'a atar."""
        return await asyncio.to_thread(
            self.rerank, query, candidates, top_k, score_threshold, deduplicate_parents
        )

    def __repr__(self) -> str:
        return f"Reranker(model={self._model_name!r}, batch_size={self._batch_size})"

