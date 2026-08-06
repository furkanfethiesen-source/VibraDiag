"""
Cross-encoder tabanlı reranker.

Hybrid search sonrası aday dokümanları query ile birlikte cross-encoder'a
vererek yeniden skorlar. Skorlar sigmoid ile [0, 1] aralığına normalize edilir,
böylece threshold karşılaştırması tutarlı olur.
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
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "mps",
        batch_size: int = 32,
    ):
        logger.info(f"Reranker yükleniyor: {model_name} ({device})")
        self.model = CrossEncoder(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size


    @staticmethod
    def _sigmoid(x: float) -> float:
        """Skoru [0, 1] aralığına normalize eder."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self,
        query: str,
        candidates: dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Candidate dokümanları cross-encoder ile yeniden skorlar.

        Parameters
        ----------
        query : str
            Kullanıcı sorgusu.
        candidates : dict
            ChromaDB-style dict: ``{ids, documents, metadatas, distances}``.
        top_k : int
            Döndürülecek en iyi sonuç sayısı.

        Returns
        -------
        dict
            Aynı ChromaDB-style dict. ``distances`` alanı artık sigmoid ile
            normalize edilmiş cross-encoder skorlarını içerir ([0, 1]).
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

        pairs = [(query, str(doc)) for doc in documents]
        raw_scores: Any = self.model.predict(pairs, batch_size=self._batch_size)

        scored = []
        for i, score in enumerate(raw_scores):
            scored.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": self._sigmoid(float(score)),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        logger.debug(
            f"Rerank tamamlandı: {len(ids)} aday → {len(top)} sonuç | "
            f"max_score={top[0]['score']:.3f}" if top else "Rerank: boş sonuç"
        )

        return {
            "ids": [[item["id"] for item in top]],
            "documents": [[item["document"] for item in top]],
            "metadatas": [[item["metadata"] for item in top]],
            "distances": [[item["score"] for item in top]],
        }

    async def arerank(
        self,
        query: str,
        candidates: dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Rerank'ın asenkron wrapper'ı — CPU-bound işlemi thread pool'a atar."""
        return await asyncio.to_thread(self.rerank, query, candidates, top_k)

    def __repr__(self) -> str:
        return f"Reranker(model={self._model_name!r}, batch_size={self._batch_size})"

