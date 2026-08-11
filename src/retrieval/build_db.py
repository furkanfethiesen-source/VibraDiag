#!/usr/bin/env python3
"""
Vector Database Build Script
============================
Proje kök dizininde bulunan, kalıcı veritabanlarını (TextChildQdrantDB, VisualQdrantDB ve DocStore)
oluşturan ve işlenmiş chunk'larla dolduran betik.

Kullanım:
---------
1. Normal çalıştırma (sadece veritabanı boşsa doldurur):
python build_db.py

2. Zorunlu yeniden oluşturma (mevcut veritabanını temizleyip sıfırdan doldurur):
python build_db.py --force
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger
from config_loader import load_appcfg, load_retcfg
from retrieval import (
    DocStore,
    Embedder,
    TextChildQdrantDB,
    VisualQdrantDB,
)
from retrieval.sparse_encoder import SparseEncoder
from schemas.schemas import VisualChunk


def get_db_paths() -> dict[str, str]:
    """Konfigürasyon dosyalarından (app.yaml & retrieval.yaml) veritabanı ve dosya yollarını okur."""
    try:
        app_cfg = load_appcfg()
        ret_cfg = load_retcfg()
        paths_cfg = getattr(app_cfg, "paths", {}) or {}
        qdrant_cfg = getattr(ret_cfg, "qdrant", {}) or {}
        docstore_cfg = getattr(ret_cfg, "docstore", {}) or {}
        return {
            "qdrant_persist_path": qdrant_cfg.get("persist_path") or paths_cfg.get("qdrant_persist_path") or "./qdrant_data",
            "docstore_path": docstore_cfg.get("persist_path") or paths_cfg.get("docstore_path") or "./docstore.db",
            "text_chunks_json": paths_cfg.get("text_chunks_json") or "./data/processed/text_chunks.json",
            "visual_chunks_json": paths_cfg.get("visual_chunks_json") or "./data/processed/visual_chunks.json",
            "raw_pdf_path": paths_cfg.get("raw_pdf_path") or "./data/raw/VibrationAnalysisandDiagnosticGuide.pdf",
        }
    except Exception:
        return {
            "qdrant_persist_path": "./qdrant_data",
            "docstore_path": "./docstore.db",
            "text_chunks_json": "./data/processed/text_chunks.json",
            "visual_chunks_json": "./data/processed/visual_chunks.json",
            "raw_pdf_path": "./data/raw/VibrationAnalysisandDiagnosticGuide.pdf",
        }


def build_vector_databases(force: bool = False) -> None:
    """
    Vektör veritabanlarını ve DocStore'u oluşturur ve işlenmiş verilerle doldurur.
    
    Parameters
    ----------
    force : bool
        True ise veritabanı boş olmasa bile sıfırdan yeniden oluşturulur.
    """
    db_paths = get_db_paths()
    qdrant_persist_path = db_paths["qdrant_persist_path"]
    docstore_path = db_paths["docstore_path"]
    text_chunks_json = db_paths["text_chunks_json"]
    visual_chunks_json = db_paths["visual_chunks_json"]
    pdf_path = db_paths["raw_pdf_path"]

    try:
        app_cfg = load_appcfg()
        embed_model = getattr(app_cfg, "embeddings", {}).get("model", "BAAI/bge-large-en-v1.5")
        device = getattr(app_cfg, "embeddings", {}).get("device", "mps")
    except Exception:
        embed_model = "BAAI/bge-large-en-v1.5"
        device = "mps"

    logger.info(f"Embedder başlatılıyor: {embed_model} ({device})")
    embedder = Embedder(model_name=embed_model, device=device)

    if force and os.path.exists(qdrant_persist_path):
        logger.warning(f"Force modu: Qdrant veritabanı siliniyor: {qdrant_persist_path}")
        shutil.rmtree(qdrant_persist_path)

    logger.info("SparseEncoder başlatılıyor...")
    sparse_encoder = SparseEncoder()

    from qdrant_client import QdrantClient

    qdrant_client_inst = QdrantClient(path=qdrant_persist_path)
    docstore = DocStore(persist_path=docstore_path)
    text_vector_db = TextChildQdrantDB(
        persist_path=qdrant_persist_path,
        sparse_encoder=sparse_encoder,
        client=qdrant_client_inst,
    )
    visual_vector_db = VisualQdrantDB(
        persist_path=qdrant_persist_path,
        client=qdrant_client_inst,
    )

    text_count = text_vector_db.count()
    visual_count = visual_vector_db.count()
    docstore_count = docstore.count()

    if not force and text_count > 0 and visual_count > 0 and docstore_count > 0:
        logger.info(
            f"Vektör veritabanı zaten dolu ve hazır. "
            f"(Text: {text_count}, Visual: {visual_count}, Parent: {docstore_count})"
        )
        logger.info("Yeniden doldurmak için '--force' bayrağını kullanabilirsiniz.")
        return

    logger.info("Kalıcı vektör veritabanı hazırlanıyor ve dolduruluyor...")

    if force or text_count == 0 or docstore_count == 0:
        if os.path.exists(text_chunks_json):
            logger.info(f"İşlenmiş metin chunk'ları okunuyor: {text_chunks_json}")
            with open(text_chunks_json, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            logger.info(f"PDF'ten metin chunk'ları çıkarılıyor: {pdf_path}")
            from ingestion.text_parser import build_chunks
            raw_chunks = build_chunks(pdf_path)
            chunks = [c.__dict__ for c in raw_chunks]

        parents = [c for c in chunks if c.get("chunk_level") == "parent"]
        children = [c for c in chunks if c.get("chunk_level") == "child"]

        if parents and (force or docstore_count == 0):
            docstore.add([p["chunk_id"] for p in parents], [p["text"] for p in parents])
            logger.info(f"DocStore'a {len(parents)} parent chunk kaydedildi.")

        if children and (force or text_count == 0):
            c_ids = [c["chunk_id"] for c in children]
            c_docs = [c["text"] for c in children]
            c_metas = [
                {
                    "parent_id": c.get("parent_id") or "",
                    "page_number": c.get("page_number", 0),
                    "section_path": c.get("section_path", "unknown"),
                    "content_type": c.get("content_type", "text"),
                }
                for c in children
            ]
            embed_texts = [
                f"{c.get('section_path', '')}\n\n{c['text']}" if c.get("section_path") else c["text"]
                for c in children
            ]
            logger.info(f"{len(children)} child chunk için embedding üretiliyor...")
            embeddings = embedder.embed_documents(embed_texts)
            text_vector_db.add(ids=c_ids, embeddings=embeddings, documents=c_docs, metadatas=c_metas)
            logger.info(f"TextChildQdrantDB'ye {len(children)} child chunk kaydedildi.")

    if (force or visual_count == 0) and os.path.exists(visual_chunks_json):
        logger.info(f"İşlenmiş görsel chunk'ları okunuyor: {visual_chunks_json}")
        with open(visual_chunks_json, "r", encoding="utf-8") as f:
            v_chunks_raw = json.load(f)

        v_ids, v_docs, v_metas, v_embed_texts = [], [], [], []
        for vc_dict in v_chunks_raw:
            vc = VisualChunk(**vc_dict)
            v_ids.append(vc.chunk_id)
            v_docs.append(vc.generated_text)
            v_metas.append(vc.to_metadata())
            v_embed_texts.append(embedder.visual_chunk_to_embedding_text(vc))

        if v_ids:
            logger.info(f"{len(v_ids)} visual chunk için embedding üretiliyor...")
            v_embeddings = embedder.embed_documents(v_embed_texts)
            visual_vector_db.add(ids=v_ids, embeddings=v_embeddings, documents=v_docs, metadatas=v_metas)
            logger.info(f"VisualQdrantDB'ye {len(v_ids)} görsel chunk kaydedildi.")

    logger.info("Vektör veritabanı başarıyla oluşturuldu ve diske kaydedildi.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibraDiag Veritabanı Oluşturma ve Doldurma Betiği",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Mevcut veritabanını sıfırlayıp sıfırdan yeniden doldurmayı zorlar",
    )
    args = parser.parse_args()
    build_vector_databases(force=args.force)


if __name__ == "__main__":
    main()
