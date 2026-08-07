"""
Dual retriever architecture + LangGraph retrieval orchestrator.

TextRetriever  — Qdrant Native Hybrid Search (Dense + Sparse) → Reranker → Parent Chunk çözümleme
VisualRetriever — Dense vector search, reranker yok
RetrievalGraph — LangGraph node olarak koşullu yönlendirme + tekilleştirme + etiketleme
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from retrieval.docstore import DocStore
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from schemas.states import RetrievalState

if TYPE_CHECKING:
    from retrieval.qdrant_vector_db import TextChildQdrantDB, VisualQdrantDB
    from retrieval.sparse_encoder import SparseEncoder


class TextRetriever:
    """
    Text child chunk'lar arasında Qdrant Native Hybrid Search (Dense + Sparse) yapar,
    cross-encoder Reranker ile yeniden skorlar, parent chunk'ları çözümler.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_db: TextChildQdrantDB,
        docstore: DocStore,
        reranker: Reranker,
        sparse_encoder: SparseEncoder | None = None,
    ):
        self.embedder = embedder
        self.vector_db = vector_db
        self.docstore = docstore
        self.reranker = reranker
        self.sparse_encoder = sparse_encoder

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Qdrant native hybrid search ile arama yapar.

        Sparse encoder mevcutsa hem dense hem sparse vektörlerle hibrit arama yapılır
        ve sonuçlar Qdrant'ın native RRF fusion'ı ile birleştirilir.
        Sparse encoder yoksa sadece dense arama yapılır.
        """
        query_embedding = self.embedder.embed_query(query)

        sparse_query = None
        if self.sparse_encoder is not None:
            sparse_query = self.sparse_encoder.encode_query(query)

        return self.vector_db.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
            sparse_query_embedding=sparse_query,
        )

    def search_with_rerank(
        self,
        query: str,
        rerank_top_k: int = 25,
        final_top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Hybrid search → Reranker → Parent chunk çözümleme.

        Dönen dict'te ``max_score`` anahtarı ek olarak bulunur (threshold
        kontrolü için). Skorlar sigmoid ile [0, 1] aralığındadır.
        """
        candidates = self.search(query, n_results=rerank_top_k, where=where)

        reranked = self.reranker.rerank(query, candidates, top_k=final_top_k)

        distances = reranked.get("distances", [[]])[0]
        max_score = max(distances) if distances else 0.0

        if not reranked.get("metadatas") or not reranked["metadatas"][0]:
            result: dict[str, Any] = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "max_score": 0.0,
            }
            return result

        parent_ids = {
            m.get("parent_id") or m.get("parent_doc_id")
            for m in reranked["metadatas"][0]
            if m and (m.get("parent_id") or m.get("parent_doc_id"))
        }

        parent_texts = self.docstore.get_many(list(parent_ids)) if parent_ids else {}
        merged = self._merge(reranked, parent_texts)
        merged["max_score"] = max_score
        return merged

    def _merge(
        self, child_results: dict[str, Any], parent_texts: dict[str, str]
    ) -> dict[str, Any]:
        """Child sonuçlarını parent metinleriyle birleştirir."""
        merged: dict[str, Any] = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        ids_list = child_results["ids"][0]
        docs_list = child_results["documents"][0]
        meta_list = child_results["metadatas"][0]
        dist_list = (
            child_results.get("distances", [[]])[0]
            if child_results.get("distances")
            else [0.0] * len(ids_list)
        )

        for i, child_id in enumerate(ids_list):
            child_metadata = meta_list[i] or {}
            child_doc = docs_list[i]
            child_distance = dist_list[i] if i < len(dist_list) else 0.0

            parent_id = child_metadata.get("parent_id") or child_metadata.get(
                "parent_doc_id"
            )
            parent_doc = parent_texts.get(parent_id) if parent_id else None

            merged_doc = (
                f"[PARENT]\n{parent_doc}\n\n[CHILD]\n{child_doc}"
                if parent_doc
                else child_doc
            )

            merged["ids"][0].append(child_id)
            merged["documents"][0].append(merged_doc)
            merged["metadatas"][0].append(child_metadata)
            merged["distances"][0].append(child_distance)

        return merged

class VisualRetriever:
    """
    Visual chunk'lar arasında dense vector search yapar.
    Reranker kullanmaz — doğrudan en iyi sonuçları döndürür.
    """

    def __init__(self, embedder: Embedder, vector_db: VisualQdrantDB):
        self.embedder = embedder
        self.vector_db = vector_db

    def search(
        self,
        query: str,
        n_results: int = 3,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dense vector search ile visual chunk'ları getirir."""
        query_embedding = self.embedder.embed_query(query)
        return self.vector_db.query(
            query_embedding=query_embedding, n_results=n_results, where=where
        )





class RetrievalGraph:
    """
    LangGraph tabanlı retrieval orkestratörü.

    TextRetriever ve VisualRetriever arasında koşullu yönlendirme yapar:
    - Text sonuçlarının ``max_score``'u threshold altındaysa VisualRetriever tetiklenir.
    - Sonuçlar ``section_path/page_number`` veya ``fault_type`` çakışmasına göre tekilleştirilir.
    - Metadata eksikse Jaccard metin benzerliği ile ikincil tekilleştirme uygulanır.
    - Etiketli bölümler (text_passages / visual_evidence) halinde döner.

    NOT: Bu sınıf LLM prompt'u oluşturmaz — sadece etiketli sonuçları yapılandırır.
    Prompt assembly ayrı bir katmanda gerçekleştirilmelidir.
    """

    JACCARD_THRESHOLD = 0.7

    def __init__(
        self,
        text_retriever: TextRetriever,
        visual_retriever: VisualRetriever,
        threshold: float = 0.35,
    ):
        self.text_retriever = text_retriever
        self.visual_retriever = visual_retriever
        self.threshold = threshold
        self.graph = self._build_graph()


    def _build_graph(self) -> Any:
        """Retrieval LangGraph'ı oluşturur ve derler."""
        workflow = StateGraph(RetrievalState)

        workflow.add_node("text_retrieve", self._text_retrieve_node)
        workflow.add_node("visual_retrieve", self._visual_retrieve_node)
        workflow.add_node("deduplicate_and_label", self._deduplicate_and_label_node)

        workflow.add_edge(START, "text_retrieve")
        workflow.add_conditional_edges(
            "text_retrieve",
            self._should_fallback,
            {
                "visual_retrieve": "visual_retrieve",
                "deduplicate_and_label": "deduplicate_and_label",
            },
        )
        workflow.add_edge("visual_retrieve", "deduplicate_and_label")
        workflow.add_edge("deduplicate_and_label", END)

        return workflow.compile()


    def _text_retrieve_node(self, state: RetrievalState) -> dict:
        """Text retrieval node: hybrid search → reranker → parent resolution."""
        results = self.text_retriever.search_with_rerank(
            query=state["query"],
            final_top_k=state.get("text_top_k", 5),
        )
        max_score = results.pop("max_score", 0.0)
        n_results = len(results.get("ids", [[]])[0])
        logger.info(f"TextRetriever: {n_results} sonuç, max_score={max_score:.3f}")
        return {
            "text_results": results,
            "max_text_score": max_score,
        }

    def _should_fallback(self, state: RetrievalState) -> str:
        """Threshold altındaysa visual retrieval'a yönlendir."""
        threshold = state.get("threshold", self.threshold)
        max_score = state.get("max_text_score", 0.0)

        if max_score < threshold:
            logger.info(
                f"Visual fallback tetiklendi: max_score ({max_score:.3f}) < threshold ({threshold})"
            )
            return "visual_retrieve"

        logger.info(
            f"Sadece text sonuçları yeterli: max_score ({max_score:.3f}) ≥ threshold ({threshold})"
        )
        return "deduplicate_and_label"

    def _visual_retrieve_node(self, state: RetrievalState) -> dict:
        """Visual retrieval node: dense vector search."""
        results = self.visual_retriever.search(
            query=state["query"],
            n_results=state.get("visual_top_k", 3),
        )
        logger.info(f"VisualRetriever: {len(results.get('ids', [[]])[0])} sonuç")
        return {
            "visual_results": results,
            "visual_triggered": True,
        }

    def _deduplicate_and_label_node(self, state: RetrievalState) -> dict:
        """
        Sonuçları tekilleştir ve etiketli passage/evidence listelerine dönüştür.

        Tekilleştirme kuralı:
        - ``section_path + page_number`` çakışması VEYA
        - ``fault_type`` çakışması VEYA
        - Metadata eksikse (section_path unknown/boş), Jaccard metin benzerliği
        varsa, visual sonuç çıkarılır (text daha zengin bağlam içerir).
        """
        text_results = state.get("text_results", {})
        visual_results = state.get("visual_results")

        text_passages = self._extract_passages(text_results)

        visual_evidence: list[dict[str, Any]] = []
        if visual_results:
            visual_evidence = self._extract_passages(visual_results)

        if text_passages and visual_evidence:
            visual_evidence = self._deduplicate(text_passages, visual_evidence)

        return {
            "text_passages": text_passages,
            "visual_evidence": visual_evidence,
            "visual_triggered": state.get("visual_triggered", False),
        }


    @staticmethod
    def _extract_passages(results: dict[str, Any]) -> list[dict[str, Any]]:
        """ChromaDB-style dict'ten passage listesi çıkarır."""
        passages: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            passages.append(
                {
                    "id": doc_id,
                    "text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": distances[i] if i < len(distances) else 0.0,
                }
            )
        return passages

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """
        İki metin arasındaki Jaccard benzerliğini hesaplar.

        Token bazlı: set(words_a) ∩ set(words_b) / set(words_a) ∪ set(words_b)
        """
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    @classmethod
    def _deduplicate(
        cls,
        text_passages: list[dict[str, Any]],
        visual_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Text ile çakışan visual sonuçları filtreler.

        Çakışma koşulu (VEYA):
        1. Aynı ``(section_path, page_number)`` çifti
        2. Aynı ``fault_type`` değeri
        3. Metadata eksikse (section_path boş/unknown), Jaccard metin benzerliği
        eşik değerinin üzerindeyse (yedek kural)
        """
        text_location_keys: set[tuple[str, str]] = set()
        text_fault_keys: set[str] = set()

        for passage in text_passages:
            meta = passage.get("metadata", {})
            loc_key = (
                meta.get("section_path", ""),
                str(meta.get("page_number", "")),
            )
            if loc_key != ("", ""):
                text_location_keys.add(loc_key)

            fault = meta.get("fault_type", "")
            if fault:
                text_fault_keys.add(fault)

        filtered: list[dict[str, Any]] = []
        for evidence in visual_evidence:
            meta = evidence.get("metadata", {})
            loc_key = (
                meta.get("section_path", ""),
                str(meta.get("page_number", "")),
            )
            fault = meta.get("fault_type", "")
            section_path = meta.get("section_path", "")

            location_collision = loc_key in text_location_keys and loc_key != ("", "")
            fault_collision = fault in text_fault_keys and fault != ""

            if location_collision or fault_collision:
                logger.debug(
                    f"Tekilleştirme: visual '{evidence.get('id')}' çıkarıldı "
                    f"(loc={location_collision}, fault={fault_collision})"
                )
                continue

            if not section_path or section_path == "unknown":
                evidence_text = evidence.get("text", "")
                is_duplicate = False
                for text_passage in text_passages:
                    similarity = cls._jaccard_similarity(
                        evidence_text, text_passage.get("text", "")
                    )
                    if similarity >= cls.JACCARD_THRESHOLD:
                        logger.debug(
                            f"Tekilleştirme (Jaccard): visual '{evidence.get('id')}' çıkarıldı "
                            f"(similarity={similarity:.3f})"
                        )
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue

            filtered.append(evidence)

        if len(visual_evidence) != len(filtered):
            logger.info(
                f"Tekilleştirme: {len(visual_evidence)} → {len(filtered)} visual sonuç"
            )

        return filtered


    def invoke(
        self,
        query: str,
        text_top_k: int = 5,
        visual_top_k: int = 3,
        threshold: float | None = None,
    ) -> RetrievalState:
        """
        Retrieval pipeline'ını çalıştırır.

        Parameters
        ----------
        query : str
            Kullanıcı sorgusu.
        text_top_k : int
            TextRetriever'dan döndürülecek nihai sonuç sayısı.
        visual_top_k : int
            VisualRetriever'dan döndürülecek sonuç sayısı.
        threshold : float | None
            Visual fallback eşik değeri. None ise constructor'daki değer kullanılır.

        Returns
        -------
        RetrievalState
            ``text_passages``, ``visual_evidence``, ``visual_triggered`` anahtarlarını içerir.
        """
        initial_state: RetrievalState = {
            "query": query,
            "text_top_k": text_top_k,
            "visual_top_k": visual_top_k,
            "threshold": threshold if threshold is not None else self.threshold,
            "text_results": {},
            "visual_results": {},
            "max_text_score": 0.0,
            "visual_triggered": False,
            "text_passages": [],
            "visual_evidence": [],
        }
        return self.graph.invoke(initial_state)
