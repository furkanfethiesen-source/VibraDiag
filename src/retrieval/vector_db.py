"""
Vector database abstraction layer.

BaseVectorDB (ABC) — Tüm implementasyonların uyması gereken sözleşme.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast


class BaseVectorDB(ABC):
    """Tüm VectorDB implementasyonlarının uyması gereken sözleşme."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Dokümanları vektörleriyle birlikte veritabanına ekler."""

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Vektör benzerliğine göre sorgu yapar. Standardize retrieval dict döner."""

    @abstractmethod
    def count(self) -> int:
        """Collection'daki toplam doküman sayısını döner."""

    @abstractmethod
    def get_all(self) -> dict[str, Any]:
        """Collection'daki tüm dokümanları döner."""
