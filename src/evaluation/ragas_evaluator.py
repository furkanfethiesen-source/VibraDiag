"""
VibraDiag Retrieval Evaluation Pipeline (RAGAS & Deterministik Metrikler)
=======================================================================
`RetrievalGraph`'ın uçtan uca çıktısını (TextRetriever hybrid search + rerank
+ visual fallback + dedup/label) `retrieval_benchmark_dev8.json` veya
`retrieval_benchmark.json` altın (gold) veri seti üzerinde hem deterministik
ID-bazlı metriklerle hem de RAGAS metrikleriyle değerlendirir.

Deterministik Metrikler:
- Child-level Recall@K, Precision@K, HitRate@K
- Parent-level Recall@K, Precision@K, HitRate@K
- Visual-level Recall@K, Precision@K

RAGAS Yargıç Metrikleri:
- Faithfulness (halüsinasyon kontrolü)
- ResponseRelevancy (cevap-sorgu alakası)
- ContextPrecision (LLMContextPrecisionWithReference)
- ContextRecall (LLMContextRecall)
- ContextEntityRecall (opsiyonel, --with-entity-recall)

Kullanım:
---------
export GOOGLE_API_KEY=...

# Hızlı deneme (ilk 3 soru, generation atla, deduplicate_parents=False)
uv run python src/evaluation/ragas_evaluator.py --sample 3 --skip-generation

# Tam değerlendirme koşumu
uv run python src/evaluation/ragas_evaluator.py --gold data/evaluation/retrieval_benchmark_dev8.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger

from config_loader import load_appcfg, load_retcfg
from evaluation.custom_metrics import (
    extract_parent_id,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evaluation.dataset import EvaluationDatasetLoader, GoldItemSchema
from evaluation.ragas_compat import apply_ragas_compatibility_patch
from generation.client import GroqClient
from generation.prompt_builder import (
    build_messages_payload,
    build_system_prompt,
    build_user_message,
)
from retrieval import DocStore, Embedder, TextChildQdrantDB, VisualQdrantDB
from retrieval.reranker import Reranker
from retrieval.retriever import RetrievalGraph, TextRetriever, VisualRetriever
from retrieval.sparse_encoder import SparseEncoder

from datasets import Dataset

apply_ragas_compatibility_patch()

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextEntityRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

try:
    from ragas.metrics import ResponseRelevancy
except ImportError:
    from ragas.metrics import AnswerRelevancy as ResponseRelevancy


def get_eval_paths() -> dict[str, str]:
    """App ve Retrieval config dosyalarından dosya ve veritabanı yollarını yükler."""
    try:
        app_cfg = load_appcfg()
        ret_cfg = load_retcfg()
        paths_cfg = getattr(app_cfg, "paths", {}) or {}
        qdrant_cfg = getattr(ret_cfg, "qdrant", {}) or {}
        docstore_cfg = getattr(ret_cfg, "docstore", {}) or {}

        qdrant_path = qdrant_cfg.get("persist_path") or paths_cfg.get("qdrant_persist_path") or "./qdrant_data"
        docstore_path = docstore_cfg.get("persist_path") or paths_cfg.get("docstore_path") or "./docstore.db"
        gold_path = (
            paths_cfg.get("gold_dataset_path")
            or ("./data/evaluation/retrieval_benchmark_dev8.json" if Path("./data/evaluation/retrieval_benchmark_dev8.json").exists() else "./retrieval_benchmark_dev8.json")
        )
        output_dir = paths_cfg.get("eval_output_dir") or "./data/eval_reports"
        return {
            "qdrant_persist_path": qdrant_path,
            "docstore_path": docstore_path,
            "gold_dataset_path": gold_path,
            "eval_output_dir": output_dir,
        }
    except Exception:
        return {
            "qdrant_persist_path": "./qdrant_data",
            "docstore_path": "./docstore.db",
            "gold_dataset_path": "./data/evaluation/retrieval_benchmark_dev8.json" if Path("./data/evaluation/retrieval_benchmark_dev8.json").exists() else "./retrieval_benchmark_dev8.json",
            "eval_output_dir": "./data/eval_reports",
        }


_EVAL_PATHS = get_eval_paths()
QDRANT_PERSIST_PATH = _EVAL_PATHS["qdrant_persist_path"]
DOCSTORE_PATH = _EVAL_PATHS["docstore_path"]
DEFAULT_GOLD_PATH = _EVAL_PATHS["gold_dataset_path"]
DEFAULT_OUTPUT_DIR = _EVAL_PATHS["eval_output_dir"]

_NON_METRIC_ROW_FIELDS = ("user_input", "retrieved_contexts", "reference", "response", "_id")


def build_retrieval_graph(enable_decomposer: bool = False) -> tuple[RetrievalGraph, dict[str, Any]]:
    """
    Kalıcı Qdrant/DocStore veritabanlarına bağlanarak tam RetrievalGraph'ı kurar.
    Konfigürasyonu yükler ve açıkça çözümlenen parametreleri döner.
    """
    try:
        app_cfg = load_appcfg()
        ret_cfg = load_retcfg()
        paths_cfg = getattr(app_cfg, "paths", {}) or {}
        qdrant_cfg = getattr(ret_cfg, "qdrant", {}) or {}
        docstore_cfg = getattr(ret_cfg, "docstore", {}) or {}

        qdrant_path = qdrant_cfg.get("persist_path") or paths_cfg.get("qdrant_persist_path") or QDRANT_PERSIST_PATH
        docstore_path = docstore_cfg.get("persist_path") or paths_cfg.get("docstore_path") or DOCSTORE_PATH
        embed_model = getattr(app_cfg, "embeddings", {}).get("model", "BAAI/bge-m3")
        device = getattr(app_cfg, "embeddings", {}).get("device", "mps")
        reranker_cfg = getattr(ret_cfg, "hybrid_search", {}).get("reranker", {}) or {}
        reranker_model = reranker_cfg.get("model", "BAAI/bge-reranker-large")
        reranker_device = reranker_cfg.get("device", device)
        reranker_score_threshold = float(reranker_cfg.get("score_threshold", 0.20))
        reranker_floor = float(reranker_cfg.get("soft_fallback_floor", 0.10))
        visual_fallback_threshold = float(reranker_cfg.get("visual_fallback_threshold", 0.20))
        config_source = "retrieval.yaml + app.yaml"
    except Exception as err:
        logger.error(f"Konfigürasyon dosyaları yüklenirken hata oluştu: {err}. Varsayılan değerler kullanılıyor.")
        qdrant_path = QDRANT_PERSIST_PATH
        docstore_path = DOCSTORE_PATH
        embed_model = "BAAI/bge-m3"
        device = "mps"
        reranker_model = "BAAI/bge-reranker-large"
        reranker_device = "mps"
        reranker_score_threshold = 0.20
        reranker_floor = 0.10
        visual_fallback_threshold = 0.20
        config_source = f"FALLBACK_DEFAULTS (hata: {err})"

    logger.info(f"Embedder başlatılıyor: {embed_model} ({device}) [Kaynak: {config_source}]")
    embedder = Embedder(model_name=embed_model, device=device)

    logger.info("SparseEncoder başlatılıyor...")
    sparse_encoder = SparseEncoder()

    logger.info(f"Reranker başlatılıyor: {reranker_model} ({reranker_device}) (threshold={reranker_score_threshold:.2f}, floor={reranker_floor:.2f})")
    reranker = Reranker(
        model_name=reranker_model,
        device=reranker_device,
        score_threshold=reranker_score_threshold,
        soft_fallback_floor=reranker_floor,
    )

    from qdrant_client import QdrantClient

    qdrant_client_inst = QdrantClient(path=qdrant_path)
    docstore = DocStore(persist_path=docstore_path)
    text_vector_db = TextChildQdrantDB(
        persist_path=qdrant_path,
        sparse_encoder=sparse_encoder,
        client=qdrant_client_inst,
    )
    visual_vector_db = VisualQdrantDB(
        persist_path=qdrant_path,
        client=qdrant_client_inst,
    )

    if text_vector_db.count() == 0:
        logger.warning("TextChildQdrantDB boş görünüyor — önce `python build_db.py` çalıştırıldığından emin olun.")

    text_retriever = TextRetriever(
        embedder=embedder,
        vector_db=text_vector_db,
        docstore=docstore,
        reranker=reranker,
        sparse_encoder=sparse_encoder,
    )
    visual_retriever = VisualRetriever(embedder=embedder, vector_db=visual_vector_db)

    classifier = None
    decomposer = None
    if enable_decomposer:
        try:
            from query_decomposer.classifier import QueryComplexityClassifier
            from query_decomposer.llm_decomposer import LLMDecomposer
            classifier = QueryComplexityClassifier(
                model_path="src/query_decomposer/models/complexity_pipeline.pkl",
                training_data_path="data/decomposer_data/training.jsonl",
            )
            decomposer = LLMDecomposer()
            logger.info("Decomposer ve QueryComplexityClassifier aktif.")
        except Exception as e:
            logger.warning(f"Decomposer başlatılamadı: {e}")

    graph = RetrievalGraph(
        text_retriever=text_retriever,
        visual_retriever=visual_retriever,
        threshold=visual_fallback_threshold,
        classifier=classifier,
        decomposer=decomposer,
    )

    params = {
        "config_source": config_source,
        "decomposer_enabled": enable_decomposer,
        "embedder_model": embed_model,
        "embedder_device": device,
        "sparse_encoder_model": sparse_encoder.model.__class__.__name__ if hasattr(sparse_encoder, "model") else "n/a",
        "reranker_model": reranker_model,
        "reranker_device": reranker_device,
        "reranker_candidate_pool_rerank_top_k": 25,
        "reranker_score_threshold": reranker.score_threshold,
        "reranker_soft_fallback_floor": reranker.soft_fallback_floor,
        "visual_fallback_threshold": visual_fallback_threshold,
    }

    return graph, params


def build_generation_client(skip: bool, model_name: str | None = None) -> tuple[GroqClient | None, dict[str, Any]]:
    """GroqClient'ı kurar ve raporlanacak generator parametrelerini döner."""
    if skip:
        return None, {
            "generator_enabled": False,
            "generator_model": "n/a (--skip-generation)",
            "generator_temperature": "n/a",
            "generator_max_tokens": "n/a",
        }

    cfg = {"model": model_name} if model_name else None
    client = GroqClient(config=cfg)
    params = {
        "generator_enabled": True,
        "generator_model": client.model,
        "generator_temperature": client.temperature,
        "generator_max_tokens": client.max_tokens,
    }
    return client, params


def load_gold_dataset(path: str) -> list[dict[str, Any]]:
    loader = EvaluationDatasetLoader(path)
    validated_items = loader.load_items()
    logger.info(f"{len(validated_items)} adet şema doğrulanmış altın soru yüklendi: {path}")
    return [item.to_dict() for item in validated_items]


def _generate_answer(client: GroqClient, question: str, contexts: list[str]) -> str:
    """Retrieved context'lerle basit bir RAG QA cevabı üretir."""
    merged_context = "\n\n".join(contexts)
    system_prompt = build_system_prompt(has_signal_context=False, state=None, is_followup=False)
    user_message = build_user_message(query=question, context=merged_context)
    messages = build_messages_payload(system_prompt=system_prompt, user_message=user_message, chat_history=None)
    return client.generate(messages=messages)


async def _run_single(
    graph: RetrievalGraph,
    generation_client: GroqClient | None,
    item: dict[str, Any],
    text_top_k: int,
    visual_top_k: int,
    threshold: float | None,
    deduplicate_parents: bool = False,
) -> dict[str, Any]:
    question = item["question"]
    q_id = item.get("_id", "unknown")
    retrieval_error = None
    generation_failed = False

    try:
        state = await graph.ainvoke(
            question,
            text_top_k=text_top_k,
            visual_top_k=visual_top_k,
            threshold=threshold,
            deduplicate_parents=deduplicate_parents,
        )
    except Exception as err:
        logger.error(f"[{q_id}] Retrieval hatası: {err}")
        retrieval_error = str(err)
        state = {
            "text_passages": [],
            "visual_evidence": [],
            "visual_triggered": False,
        }

    text_passages = state.get("text_passages", [])
    visual_evidence = state.get("visual_evidence", [])

    contexts = [p["text"] for p in text_passages if p.get("text")]
    contexts += [v["text"] for v in visual_evidence if v.get("text")]

    retrieved_text_ids = [p["id"] for p in text_passages]
    retrieved_visual_ids = [v["id"] for v in visual_evidence]
    retrieved_parent_ids = list(dict.fromkeys([extract_parent_id(p["id"], p.get("metadata")) for p in text_passages]))

    if not contexts and not retrieval_error:
        logger.warning(f"[{q_id}] Boş context döndü: '{question}'")

    generated_answer = ""
    if generation_client is not None and contexts:
        try:
            generated_answer = await asyncio.to_thread(_generate_answer, generation_client, question, contexts)
        except Exception as err:
            logger.error(f"[{q_id}] Generation hatası: {err}")
            generated_answer = f"[GENERATION HATASI: {err}]"
            generation_failed = True
    elif generation_client is not None and not contexts:
        generated_answer = "[BAĞLAM YOK: Yanıt üretilmedi]"
        generation_failed = True

    expected_child_ids = item.get("expected_child_chunk_ids") or [
        x for x in item.get("expected_chunk_ids", []) if "child" in x
    ]
    expected_parent_ids = item.get("expected_parent_chunk_ids") or [
        x for x in item.get("expected_chunk_ids", []) if "child" not in x
    ]
    expected_visual_ids = item.get("expected_visual_chunk_ids", [])

    det_child_recall = recall_at_k(retrieved_text_ids, expected_child_ids, k=text_top_k)
    det_child_precision = precision_at_k(retrieved_text_ids, expected_child_ids, k=text_top_k)
    det_child_hit_rate = hit_rate_at_k(retrieved_text_ids, expected_child_ids, k=text_top_k)
    det_child_mrr = mrr_at_k(retrieved_text_ids, expected_child_ids, k=text_top_k)
    det_child_ndcg = ndcg_at_k(retrieved_text_ids, expected_child_ids, k=text_top_k)

    det_parent_recall = recall_at_k(retrieved_parent_ids, expected_parent_ids, k=text_top_k)
    det_parent_precision = precision_at_k(retrieved_parent_ids, expected_parent_ids, k=text_top_k)
    det_parent_hit_rate = hit_rate_at_k(retrieved_parent_ids, expected_parent_ids, k=text_top_k)
    det_parent_mrr = mrr_at_k(retrieved_parent_ids, expected_parent_ids, k=text_top_k)
    det_parent_ndcg = ndcg_at_k(retrieved_parent_ids, expected_parent_ids, k=text_top_k)

    det_visual_recall = recall_at_k(retrieved_visual_ids, expected_visual_ids, k=visual_top_k) if expected_visual_ids else 0.0

    return {
        "row": {
            "_id": q_id,
            "user_input": question,
            "retrieved_contexts": contexts if contexts else ["[BOŞ BAĞLAM]"],
            "reference": item["ground_truth_answer"],
            "response": generated_answer,
        },
        "record": {
            "_id": q_id,
            "question": question,
            "fault_type": item.get("fault_type"),
            "ground_truth_answer": item["ground_truth_answer"],
            "generated_answer": generated_answer,
            "generation_failed": generation_failed,
            "retrieval_error": retrieval_error,
            "expected_chunk_ids": item.get("expected_chunk_ids", []),
            "expected_child_chunk_ids": expected_child_ids,
            "expected_parent_chunk_ids": expected_parent_ids,
            "expected_visual_chunk_ids": expected_visual_ids,
            "reranked_chunk_ids": retrieved_text_ids,
            "retrieved_parent_ids": retrieved_parent_ids,
            "retrieved_visual_ids": retrieved_visual_ids,
            "visual_triggered": state.get("visual_triggered", False),
            "n_retrieved_contexts": len(contexts),
            "det_child_recall": det_child_recall,
            "det_child_precision": det_child_precision,
            "det_child_hit_rate": det_child_hit_rate,
            "det_child_mrr": det_child_mrr,
            "det_child_ndcg": det_child_ndcg,
            "det_parent_recall": det_parent_recall,
            "det_parent_precision": det_parent_precision,
            "det_parent_hit_rate": det_parent_hit_rate,
            "det_parent_mrr": det_parent_mrr,
            "det_parent_ndcg": det_parent_ndcg,
            "det_visual_recall": det_visual_recall,
        },
    }


async def build_ragas_dataset(
    graph: RetrievalGraph,
    generation_client: GroqClient | None,
    gold_items: list[dict[str, Any]],
    text_top_k: int,
    visual_top_k: int,
    threshold: float | None,
    deduplicate_parents: bool = False,
    concurrency: int = 4,
) -> tuple[Dataset, list[dict[str, Any]]]:
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _run_single(
                graph,
                generation_client,
                item,
                text_top_k,
                visual_top_k,
                threshold,
                deduplicate_parents=deduplicate_parents,
            )

    results = await asyncio.gather(*(_bounded(item) for item in gold_items))
    rows = [r["row"] for r in results]
    records = [r["record"] for r in results]
    return Dataset.from_list(rows), records


def get_ragas_llm(model_name: str = "gemini-3.1-flash-lite") -> Any:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY veya GEMINI_API_KEY ortam değişkeni tanımlı değil. .env dosyasını kontrol edin.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    chat = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
    )
    return LangchainLLMWrapper(chat)


def get_ragas_embeddings(text_embedder: Embedder) -> Any:
    return LangchainEmbeddingsWrapper(text_embedder.embeddings)


def get_metrics(
    llm: Any,
    embeddings: Any,
    include_answer_metrics: bool,
    with_entity_recall: bool = False,
) -> list:
    metrics: list = []
    if include_answer_metrics:
        metrics.append(Faithfulness(llm=llm))
        metrics.append(ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1))
    metrics.append(LLMContextPrecisionWithReference(llm=llm))
    metrics.append(LLMContextRecall(llm=llm))
    if with_entity_recall:
        metrics.append(ContextEntityRecall(llm=llm))
    return metrics


def _metric_columns(df_columns: list[str]) -> list[str]:
    return [c for c in df_columns if c not in _NON_METRIC_ROW_FIELDS]


def _attach_scores_to_records(result: Any, records: list[dict[str, Any]], expected_metric_names: list[str]) -> list[str]:
    if not hasattr(result, "to_pandas"):
        return []

    df = result.to_pandas()
    metric_cols = _metric_columns(list(df.columns))

    df_map = {}
    for idx, row in df.iterrows():
        row_id = row.get("_id")
        if row_id:
            df_map[row_id] = row.to_dict()
        else:
            df_map[records[idx]["_id"]] = row.to_dict()

    for record in records:
        r_id = record["_id"]
        matched_row = df_map.get(r_id, {})
        for col in metric_cols:
            val = matched_row.get(col)
            record[col] = float(val) if val is not None and str(val) != "nan" else 0.0

    unmatched = set(expected_metric_names) - set(metric_cols)
    if unmatched:
        logger.warning(f"RAGAS değerlendirme çıktısında beklenen metrikler eksik: {unmatched}")

    return metric_cols


def _format_star_rating(score: float) -> str:
    """Metrik skoru için yıldızlı ve renkli emoji gösterimi oluşturur."""
    if score >= 0.8:
        return "🟢 ⭐⭐⭐⭐⭐"
    elif score >= 0.6:
        return "🟡 ⭐⭐⭐⭐☆"
    elif score >= 0.4:
        return "🟠 ⭐⭐⭐☆☆"
    elif score > 0.0:
        return "🔴 ⭐⭐☆☆☆"
    else:
        return "⚪ ☆☆☆☆☆"


def write_markdown_report(
    md_path: Path,
    run_id: str,
    gold_path: str,
    n_questions: int,
    params: dict[str, Any],
    aggregate: dict[str, float],
    deterministic_summary: dict[str, float],
    fault_type_summary: dict[str, dict[str, float]],
    n_no_context: int,
    n_generation_errors: int,
    n_retrieval_errors: int,
    records: list[dict[str, Any]] | None = None,
) -> None:
    lines: list[str] = []
    lines.append(f"# 📊 VibraDiag Evaluation Report — `{run_id}`")
    lines.append("")
    lines.append("![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square&logo=github)")
    lines.append(f"![Questions](https://img.shields.io/badge/Dataset-{n_questions}%20Queries-blue?style=flat-square)")
    lines.append("![Evaluator](https://img.shields.io/badge/Evaluator-RAGAS%20%2B%20Deterministic-orange?style=flat-square)")
    lines.append("")
    lines.append(f"- 📅 **Çalıştırma zamanı:** `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- 🎯 **Gold dataset:** `{gold_path}` ({n_questions} soru)")
    lines.append("")

    lines.append("## ⚙️ System & Pipeline Configuration")
    lines.append("")
    lines.append("| Kategori | Parametre | Değer |")
    lines.append("| :--- | :--- | :--- |")
    for key, value in params.items():
        category = "Retrieval"
        k_lower = key.lower()
        if "reranker" in k_lower:
            category = "Reranker"
        elif "embedder" in k_lower or "sparse" in k_lower:
            category = "Embeddings"
        elif "generator" in k_lower:
            category = "Generation"
        elif "ragas" in k_lower or "judge" in k_lower:
            category = "RAGAS Judge"
        elif "gold" in k_lower or "question" in k_lower or "dataset" in k_lower:
            category = "Dataset"
        lines.append(f"| **{category}** | `{key}` | `{value}` |")
    lines.append("")

    lines.append("## 🎯 Deterministik ID-Seviyesi Metrikler (LLM Bağımsız)")
    lines.append("")
    lines.append("| Metrik Kümeleri | Metrik | Skor | Derece |")
    lines.append("| :--- | :--- | :---: | :---: |")
    for metric, score in deterministic_summary.items():
        group = "Child Chunk" if "child" in metric else ("Parent Chunk" if "parent" in metric else "Visual Chunk")
        rating = _format_star_rating(score)
        lines.append(f"| **{group}** | `{metric}` | **{score:.4f}** | {rating} |")
    lines.append("")

    lines.append("## ✨ Aggregate RAGAS Metrikleri")
    lines.append("")
    if aggregate:
        lines.append("| Metrik Kategorisi | RAGAS Metrik | Skor | Derece |")
        lines.append("| :--- | :--- | :---: | :---: |")
        for metric, score in aggregate.items():
            m_lower = metric.lower()
            cat = "Generation" if m_lower in ("faithfulness", "answer_relevancy", "responserelevancy") else "Semantic Retrieval"
            rating = _format_star_rating(score)
            lines.append(f"| **{cat}** | `{metric}` | **{score:.4f}** | {rating} |")
    else:
        lines.append("> [!NOTE]")
        lines.append("> RAGAS LLM Judge değerlendirmesi bu koşumda atlandı (`--skip-ragas`).")
    lines.append("")

    lines.append("## 🔍 Fault Type Kırılımı")
    lines.append("")
    metric_names = sorted({m for scores in fault_type_summary.values() for m in scores})
    header = "| Fault Type | " + " | ".join(f"`{m}`" for m in metric_names) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(metric_names)) + " |"
    lines.append(header)
    lines.append(sep)
    for ft, scores in fault_type_summary.items():
        row = f"| **{ft}** | " + " | ".join(f"**{scores.get(m, 0.0):.4f}**" for m in metric_names) + " |"
        lines.append(row)
    lines.append("")

    if records:
        lines.append("## 📋 Soru Bazlı Detaylı Kıyaslama Tablosu")
        lines.append("")
        lines.append("| ID | Soru | Kategori | Hit (Parent) | Beklenen Parent | Çekilen Parent | Context Durumu |")
        lines.append("| :--- | :--- | :--- | :---: | :--- | :--- | :--- |")
        for r in records:
            qid = r.get("_id", "N/A")
            q = (r.get("question") or "")[:45] + ("..." if len(r.get("question", "")) > 45 else "")
            ft = r.get("fault_type", "genel")
            hit = "✅ Başarılı" if (r.get("det_parent_hit_rate", 0.0) or 0.0) > 0 else "❌ Kaçtı"
            exp_p = ", ".join(r.get("expected_parent_chunk_ids") or [])
            ret_p = ", ".join(r.get("retrieved_parent_ids") or [])
            ctx_stat = f"🟢 {r.get('n_retrieved_contexts', 0)} chunk" if r.get("n_retrieved_contexts", 0) > 0 else "🔴 Boş Context"
            lines.append(f"| `{qid}` | {q} | `{ft}` | **{hit}** | `{exp_p}` | `{ret_p}` | {ctx_stat} |")
        lines.append("")

    if n_no_context or n_generation_errors or n_retrieval_errors:
        lines.append("## ⚠️ Uyarılar ve Hata Özeti")
        lines.append("")
        lines.append("> [!WARNING]")
        if n_retrieval_errors:
            lines.append(f"> - 🚨 **{n_retrieval_errors} soru** için retrieval adımı istisna vererek başarısız oldu.")
        if n_no_context:
            lines.append(f"> - 🔍 **{n_no_context} soru** için hiç retrieved context bulunamadı.")
        if n_generation_errors:
            lines.append(f"> - 💬 **{n_generation_errors} soru** için generation adımı başarısız oldu veya bağlam bulunamadığı için atlandı (RAGAS süzgecinden geçirildi).")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown rapor kaydedildi: {md_path}")


def write_csv_report(
    csv_path: Path,
    records: list[dict[str, Any]],
    metric_cols: list[str],
) -> None:
    det_cols = [
        "det_child_recall",
        "det_child_precision",
        "det_child_hit_rate",
        "det_child_mrr",
        "det_child_ndcg",
        "det_parent_recall",
        "det_parent_precision",
        "det_parent_hit_rate",
        "det_parent_mrr",
        "det_parent_ndcg",
        "det_visual_recall",
    ]
    fieldnames = (
        [
            "_id",
            "fault_type",
            "question",
            "generated_answer",
            "generation_failed",
            "retrieval_error",
            "ground_truth_answer",
            "expected_child_chunk_ids",
            "expected_parent_chunk_ids",
            "expected_visual_chunk_ids",
            "reranked_chunk_ids",
            "retrieved_parent_ids",
            "retrieved_visual_ids",
            "visual_triggered",
        ]
        + det_cols
        + metric_cols
    )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for list_field in (
                "expected_chunk_ids",
                "expected_child_chunk_ids",
                "expected_parent_chunk_ids",
                "expected_visual_chunk_ids",
                "reranked_chunk_ids",
                "retrieved_parent_ids",
                "retrieved_visual_ids",
            ):
                row[list_field] = ";".join(row.get(list_field) or [])
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info(f"CSV rapor kaydedildi: {csv_path}")


def build_and_save_reports(
    result: Any,
    records: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    gold_path: str,
    params: dict[str, Any],
    metrics_used: list,
) -> None:
    expected_metric_names = [m.name for m in metrics_used]
    metric_cols = _attach_scores_to_records(result, records, expected_metric_names)

    aggregate: dict[str, float] = {}
    if hasattr(result, "to_pandas"):
        df_res = result.to_pandas()
        for col in metric_cols:
            if col in df_res.columns:
                valid_vals = df_res[col].dropna()
                if not valid_vals.empty:
                    aggregate[col] = round(float(valid_vals.mean()), 4)

    det_keys = [
        "det_child_recall",
        "det_child_precision",
        "det_child_hit_rate",
        "det_child_mrr",
        "det_child_ndcg",
        "det_parent_recall",
        "det_parent_precision",
        "det_parent_hit_rate",
        "det_parent_mrr",
        "det_parent_ndcg",
        "det_visual_recall",
    ]
    deterministic_summary = {
        k: round(sum(r.get(k, 0.0) for r in records) / len(records), 4) for k in det_keys if records
    }

    by_fault_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        ft = record.get("fault_type") or "unknown"
        by_fault_type.setdefault(ft, []).append(record)

    fault_type_summary = {
        ft: {
            metric: round(sum(item.get(metric, 0.0) or 0.0 for item in items) / len(items), 4)
            for metric in (det_keys + metric_cols)
        }
        for ft, items in by_fault_type.items()
    }

    n_no_context = sum(1 for r in records if r["n_retrieved_contexts"] == 0)
    n_generation_errors = sum(1 for r in records if r.get("generation_failed"))
    n_retrieval_errors = sum(1 for r in records if r.get("retrieval_error") is not None)

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"eval_report_{run_id}.md"
    csv_path = output_dir / f"eval_report_{run_id}.csv"

    write_markdown_report(
        md_path=md_path,
        run_id=run_id,
        gold_path=gold_path,
        n_questions=len(records),
        params=params,
        aggregate=aggregate,
        deterministic_summary=deterministic_summary,
        fault_type_summary=fault_type_summary,
        n_no_context=n_no_context,
        n_generation_errors=n_generation_errors,
        n_retrieval_errors=n_retrieval_errors,
        records=records,
    )
    write_csv_report(csv_path=csv_path, records=records, metric_cols=metric_cols)

    print("\n=== Deterministik ID-Seviyesi Sonuçlar ===")
    for k, v in deterministic_summary.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== RAGAS Değerlendirme Sonuçları ===")
    for metric, score in aggregate.items():
        print(f"  {metric}: {score:.4f}")

    print("\n=== Fault Type Kırılımı ===")
    for ft, scores in fault_type_summary.items():
        scores_str = ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
        print(f"  {ft}: {scores_str}")

    if n_no_context or n_generation_errors or n_retrieval_errors:
        print(f"\n⚠ Hata Durumu: {n_retrieval_errors} retrieval hatası, {n_no_context} boş bağlam, {n_generation_errors} üretme hatası.")

    print(f"\nRapor dosyaları:\n  {md_path}\n  {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibraDiag Retrieval RAGAS & Deterministik Değerlendirme Betiği",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gold", default=DEFAULT_GOLD_PATH, help="Gold dataset json yolu")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Rapor kaydedilecek dizin")
    parser.add_argument("--run-name", default=None, help="Rapor run id")
    parser.add_argument("--text-top-k", type=int, default=3, help="TextRetriever final_top_k")
    parser.add_argument("--visual-top-k", type=int, default=3, help="VisualRetriever top_k")
    parser.add_argument("--threshold", type=float, default=None, help="Visual fallback eşiği (verilmezse retrieval.yaml kullanılır)")
    parser.add_argument("--deduplicate-parents", action="store_true", default=False, help="Eval esnasında parent tekilleştirme yap (Varsayılan: False)")
    parser.add_argument("--enable-decomposer", action="store_true", default=False, help="Decomposer ve alt sorgu ayrıştırmasını aktif et (Varsayılan: False)")
    parser.add_argument("--enable-corrector", action="store_true", default=False, help="Self-Corrector doğrulama döngüsünü aktif et (Varsayılan: False)")
    parser.add_argument("--with-entity-recall", action="store_true", help="ContextEntityRecall metriğini ekle")
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite", help="RAGAS LLM judge modeli")
    parser.add_argument("--generator-model", default="qwen/qwen3.6-27b", help="Üretim (generation) LLM modeli")
    parser.add_argument("--skip-ragas", action="store_true", help="RAGAS LLM judge adımını atla (sadece deterministik ID metriklerini hesapla)")
    parser.add_argument("--skip-generation", action="store_true", help="generation adımını atla")
    parser.add_argument("--retrieval-concurrency", type=int, default=4, help="Yerel retrieval eşzamanlılık")
    parser.add_argument("--judge-concurrency", type=int, default=2, help="RAGAS LLM judge API eşzamanlılık")
    parser.add_argument("--max-workers", type=int, default=None, help="Geriye uyumluluk için eşzamanlılık ayarı")
    parser.add_argument("--sample", type=int, default=None, help="Yalnızca ilk N soruyu değerlendir")

    args = parser.parse_args()

    retrieval_concurrency = args.retrieval_concurrency if args.max_workers is None else args.max_workers
    judge_concurrency = args.judge_concurrency if args.max_workers is None else args.max_workers

    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    gold_items = load_gold_dataset(args.gold)
    if args.sample:
        gold_items = gold_items[: args.sample]
        logger.info(f"--sample: yalnızca ilk {len(gold_items)} soru değerlendirilecek")

    logger.info(f"Retrieval pipeline kuruluyor (enable_decomposer={args.enable_decomposer})...")
    graph, retrieval_params = build_retrieval_graph(enable_decomposer=args.enable_decomposer)

    generation_client, generation_params = build_generation_client(skip=args.skip_generation, model_name=args.generator_model)

    logger.info("RetrievalGraph (+ generation) tüm sorular için çalıştırılıyor...")
    dataset, records = asyncio.run(
        build_ragas_dataset(
            graph,
            generation_client,
            gold_items,
            text_top_k=args.text_top_k,
            visual_top_k=args.visual_top_k,
            threshold=args.threshold,
            deduplicate_parents=args.deduplicate_parents,
            concurrency=retrieval_concurrency,
        )
    )

    result = {}
    metrics = []

    if not args.skip_ragas:
        include_answer_metrics = not args.skip_generation
        if args.skip_generation:
            logger.warning("--skip-generation verildi: faithfulness ve answer_relevancy hesaplanmayacak.")

        logger.info(f"RAGAS LLM judge kuruluyor: {args.judge_model}")
        llm = get_ragas_llm(model_name=args.judge_model)
        embeddings = get_ragas_embeddings(graph.text_retriever.embedder)
        metrics = get_metrics(
            llm,
            embeddings,
            include_answer_metrics=include_answer_metrics,
            with_entity_recall=args.with_entity_recall,
        )

        eval_dataset = dataset
        if include_answer_metrics:
            valid_indices = [i for i, r in enumerate(records) if not r.get("generation_failed")]
            if len(valid_indices) < len(records):
                logger.info(f"RAGAS evaluation: {len(records) - len(valid_indices)} soru üretme hatası aldığı için süzüldü.")
                eval_dataset = dataset.select(valid_indices)

        logger.info(f"RAGAS evaluate() çalıştırılıyor ({len(metrics)} metrik, judge_concurrency={judge_concurrency})...")
        eval_ds = EvaluationDataset.from_hf_dataset(eval_dataset)
        result = evaluate(
            dataset=eval_ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=judge_concurrency, max_retries=3, timeout=60.0),
        )
    else:
        logger.info("--skip-ragas verildi: RAGAS LLM judge değerlendirmesi atlandı, sadece deterministik ID metrikleri raporlanıyor.")

    resolved_threshold = (
        args.threshold
        if args.threshold is not None
        else retrieval_params["visual_fallback_threshold"]
    )
    threshold_source = "CLI --threshold" if args.threshold is not None else "retrieval.yaml"

    all_params = {
        "gold_dataset": args.gold,
        "n_questions": len(gold_items),
        "text_top_k (final, reranker sonrası)": args.text_top_k,
        "visual_top_k": args.visual_top_k,
        "deduplicate_parents (eval)": args.deduplicate_parents,
        "visual_fallback_threshold (kullanılan)": f"{resolved_threshold} [{threshold_source}]",
        **retrieval_params,
        "ragas_judge_model": args.judge_model if not args.skip_ragas else "n/a (--skip-ragas)",
        "ragas_metrics": ", ".join(m.name for m in metrics) if metrics else "n/a (--skip-ragas)",
        **generation_params,
        "retrieval_concurrency": retrieval_concurrency,
        "judge_concurrency": judge_concurrency,
    }

    output_dir = Path(args.output_dir)
    build_and_save_reports(
        result=result,
        records=records,
        output_dir=output_dir,
        run_id=run_id,
        gold_path=args.gold,
        params=all_params,
        metrics_used=metrics,
    )


if __name__ == "__main__":
    main()