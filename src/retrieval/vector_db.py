"""
Vector database abstraction layer.

BaseVectorDB (ABC) — Tüm implementasyonların uyması gereken sözleşme.

.. deprecated::
    _ChromaVectorDB, TextChildVectorDB ve VisualVectorDB sınıfları
    Qdrant tabanlı implementasyonlar lehine kullanımdan kaldırılmıştır.
    Yeni kodda ``qdrant_vector_db.TextChildQdrantDB`` ve
    ``qdrant_vector_db.VisualQdrantDB`` kullanın.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI


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
        """Vektör benzerliğine göre sorgu yapar. ChromaDB-style dict döner."""

    @abstractmethod
    def count(self) -> int:
        """Collection'daki toplam doküman sayısını döner."""

    @abstractmethod
    def get_all(self) -> dict[str, Any]:
        """Collection'daki tüm dokümanları döner."""


class _ChromaVectorDB(BaseVectorDB):
    """
    ChromaDB PersistentClient tabanlı ortak implementasyon.
    TextChildVectorDB ve VisualVectorDB bu sınıftan türer.
    """

    def __init__(
        self,
        collection_name: str,
        persist_path: str = "./.chroma_db",
        client: ClientAPI | None = None,
    ):
        self.client = client or chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self._collection_name = collection_name

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=documents,
            metadatas=cast(Any, metadatas),
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return cast(dict[str, Any], self.collection.query(**kwargs))

    def count(self) -> int:
        return self.collection.count()

    def get_all(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.collection.get())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(collection={self._collection_name!r}, count={self.count()})"
        )

class TextChildVectorDB(_ChromaVectorDB):
    """
    Text child chunk'ları embed edip saklayan ChromaDB wrapper.
    Parent chunk'lar DocStore'da tutulur; burada yalnızca child'lar bulunur.
    """

    def __init__(
        self,
        collection_name: str = "vibra_diag",
        persist_path: str = "./.chroma_db",
        client: ClientAPI | None = None,
    ):
        super().__init__(collection_name, persist_path, client)


class VisualVectorDB(_ChromaVectorDB):
    """
    Visual chunk'ların generated_text'lerini embed edip saklayan DB.
    Kalan alanlar (content_type, page_number, section_path, fault_type, …)
    metadata olarak saklanır.
    """

    def __init__(
        self,
        collection_name: str = "vibra_diag_visual",
        persist_path: str = "./.chroma_db",
        client: ClientAPI | None = None,
    ):
        super().__init__(collection_name, persist_path, client)
