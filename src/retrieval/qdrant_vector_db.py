"""
Qdrant vector database implementation for VibraDiag.
"""

from __future__ import annotations

from typing import Any

import uuid

from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from retrieval.sparse_encoder import SparseEncoder
from retrieval.vector_db import BaseVectorDB


def _to_uuid(doc_id: str) -> str:
    """String ID'yi Qdrant uyumlu deterministik UUIDv5 formatına çevirir."""
    try:
        uuid.UUID(doc_id)
        return doc_id
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


class _QdrantVectorDB(BaseVectorDB):
    """
    Qdrant tabanlı ortak implementasyon.
    """

    def __init__(
        self,
        collection_name: str,
        vector_size: int = 1024,
        persist_path: str | None = None,
        sparse_encoder: SparseEncoder | None = None,
        client: QdrantClient | None = None,
    ):
        """
        QdrantVectorDB'yi ilklendirir.
        """
        if persist_path is None:
            try:
                from config_loader import load_appcfg, load_retcfg

                ret_cfg = load_retcfg()
                app_cfg = load_appcfg()
                persist_path = (
                    getattr(ret_cfg, "qdrant", {}).get("persist_path")
                    or getattr(app_cfg, "paths", {}).get("qdrant_persist_path")
                    or "./qdrant_data"
                )
            except Exception:
                persist_path = "./qdrant_data"

        self.collection_name = collection_name
        self.sparse_encoder = sparse_encoder
        self.client = client if client is not None else QdrantClient(path=persist_path)

        if not self.client.collection_exists(collection_name):
            logger.info(f"Koleksiyon oluşturuluyor: {collection_name}")
            
            vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)
            sparse_vectors_config = None
            
            if sparse_encoder is not None:
                sparse_vectors_config = {"sparse": SparseVectorParams()}

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="section_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="page_number",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="fault_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="content_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        else:
            logger.info(f"Var olan koleksiyona bağlanıldı: {collection_name}")

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Dokümanları vektörleriyle birlikte veritabanına ekler."""
        sparse_embeddings = None
        if self.sparse_encoder is not None:
            sparse_embeddings = self.sparse_encoder.encode(documents)

        points = []
        for i, doc_id in enumerate(ids):
            payload = dict(metadatas[i]) if metadatas else {}
            payload["_document"] = documents[i]
            payload["_original_id"] = doc_id

            vector_dict = {"": embeddings[i]}
            if sparse_embeddings is not None:
                vector_dict["sparse"] = sparse_embeddings[i]

            points.append(
                PointStruct(
                    id=_to_uuid(doc_id),
                    vector=vector_dict,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        sparse_query_embedding: SparseVector | None = None,
        weights: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Vektör benzerliğine göre sorgu yapar. Hybrid arama ve dinamik ağırlıklandırma destekler."""
        query_filter = None
        if where:
            conditions = []
            for k, v in where.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            query_filter = Filter(must=conditions)

        if self.sparse_encoder is not None and sparse_query_embedding is not None:
            dense_w, sparse_w = weights if weights is not None else (0.5, 0.5)
            dense_limit = int(n_results * (2.5 if dense_w >= sparse_w else 1.5))
            sparse_limit = int(n_results * (2.5 if sparse_w >= dense_w else 1.5))

            prefetch = [
                Prefetch(
                    query=query_embedding,
                    using="",
                    limit=dense_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_query_embedding,
                    using="sparse",
                    limit=sparse_limit,
                    filter=query_filter,
                ),
            ]
            
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=n_results,
            )
            points = response.points
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="",
                filter=query_filter,
                limit=n_results,
            )
            points = response.points

        ids = []
        documents = []
        metadatas = []
        distances = []

        for point in points:
            payload = point.payload or {}
            real_id = payload.get("_original_id") or str(point.id)
            ids.append(real_id)
            documents.append(payload.get("_document", ""))
            
            meta = {k: v for k, v in payload.items() if k not in ("_document", "_original_id")}
            metadatas.append(meta)
            
            distances.append(point.score)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def count(self) -> int:
        """Collection'daki toplam doküman sayısını döner."""
        return self.client.count(collection_name=self.collection_name).count

    def get_all(self) -> dict[str, Any]:
        """Collection'daki tüm dokümanları döner."""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
        )
        
        ids = []
        documents = []
        metadatas = []

        for point in points:
            payload = point.payload or {}
            real_id = payload.get("_original_id") or str(point.id)
            ids.append(real_id)
            documents.append(payload.get("_document", ""))
            meta = {k: v for k, v in payload.items() if k not in ("_document", "_original_id")}
            metadatas.append(meta)

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(collection={self.collection_name!r}, count={self.count()})"
        )


class TextChildQdrantDB(_QdrantVectorDB):
    """
    Text child chunk'ları embed edip saklayan Qdrant wrapper.
    Sparse encoder ile hybrid search destekler.
    """

    def __init__(
        self,
        collection_name: str = "vibra_text_child",
        vector_size: int = 1024,
        persist_path: str | None = None,
        sparse_encoder: SparseEncoder | None = None,
        client: QdrantClient | None = None,
    ):
        super().__init__(
            collection_name=collection_name,
            vector_size=vector_size,
            persist_path=persist_path,
            sparse_encoder=sparse_encoder,
            client=client,
        )


class VisualQdrantDB(_QdrantVectorDB):
    """
    Visual chunk'ların generated_text'lerini embed edip saklayan Qdrant wrapper.
    Sparse encoder kullanmaz (dense-only search).
    """

    def __init__(
        self,
        collection_name: str = "vibra_visual",
        vector_size: int = 1024,
        persist_path: str | None = None,
        client: QdrantClient | None = None,
    ):
        super().__init__(
            collection_name=collection_name,
            vector_size=vector_size,
            persist_path=persist_path,
            sparse_encoder=None,
            client=client,
        )
