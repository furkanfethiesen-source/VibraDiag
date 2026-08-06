"""
Sparse encoder for generating sparse embeddings using FastEmbed.
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client.models import SparseVector


class SparseEncoder:
    """
    FastEmbed kullanarak metinleri seyrek (sparse) vektörlere dönüştüren sınıf.
    Qdrant hybrid arama için bm42 modelini kullanır.
    """

    def __init__(self, model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"):
        """
        SparseEncoder'ı ilklendirir.

        Args:
            model_name: Kullanılacak fastembed sparse modelinin adı.
        """
        logger.info(f"SparseEncoder başlatılıyor: {model_name}")
        self.model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> list[SparseVector]:
        """
        Birden fazla metni seyrek vektörlere dönüştürür.

        Args:
            texts: Dönüştürülecek metin listesi.

        Returns:
            Qdrant SparseVector nesnelerinden oluşan liste.
        """
        embeddings = list(self.model.embed(texts))
        return [
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in embeddings
        ]

    def encode_query(self, query: str) -> SparseVector:
        """
        Tek bir sorgu metnini seyrek vektöre dönüştürür.

        Args:
            query: Dönüştürülecek sorgu metni.

        Returns:
            Qdrant SparseVector nesnesi.
        """
        embeddings = list(self.model.query_embed(query))
        emb = embeddings[0]
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
