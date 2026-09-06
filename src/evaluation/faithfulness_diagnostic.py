"""
VibraDiag — Otonom Faithfulness Diagnostik & Denetim Motoru
============================================================
`notebooks/rag_faithfulness_diagnostic_pipeline.md` tasarımına dayanan,
Markdown tablolarını, öneri maddelerini ve anlatı cümlelerini sözelize edip
çapraz dilli NLI (mDeBERTa-v3) üzerinden denetleyen otonom taksonomi motoru.

Kategori Ayrıştırması:
- Destekli: Bağlamda geçen ve doğrulanan iddialar (NLI Entailment).
- Kategori A (Format Artefaktı): Markdown tablo iskeleti, sentetik kolon başlıkları (NLI Neutral + Scaffold).
- Kategori B (Saha Çıkarımı): Prompt esnetmesiyle üretilen, bağlamı çiğnemeyen saha bakım önerileri ve teknik çıkarımlar (NLI Neutral).
- Kategori C (Gerçek Halüsinasyon): Bağlamla veya fizik kurallarıyla doğrudan çelişen hatalar (NLI Contradiction).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class ClaimExtractor:
    """RAG yanıtlarını atomik iddialara, sözelize edilmiş tablolara ve önerilere ayrıştırır."""

    DIRECTIVE_VERBS = (
        r"\b(?:önerilir|önerilmektedir|kontrol edilmelidir|yapılmalıdır|edilmelidir|"
        r"uygulanmalıdır|gerekmektedir|sağlanmalıdır|temizlenmelidir|değiştirilmelidir|"
        r"ayarlanmalıdır|incelenmelidir|seçilmelidir|alınmalıdır|kullanılmalıdır|"
        r"sağlayın|inceleyin|yapın|uygulayın|belirleyin|kontrol edin)\b"
    )

    MODAL_SUFFIXES = (
        r"(?:malı|meli|malıdır|melidir)"
    )

    @staticmethod
    def clean_markdown_inline(text: str) -> str:
        """Yıldız, link ve biçim etiketlerini temizler."""
        t = re.sub(r"\*\*([^*]+)\*\*", r"\g<1>", text)
        t = re.sub(r"\*([^*]+)\*", r"\g<1>", text)
        t = re.sub(r"`([^`]+)`", r"\g<1>", text)
        return t.strip()

    @classmethod
    def extract_tables(cls, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Metindeki Markdown tablolarını tespit eder, sözelize eder ve metinden çıkarır.
        Döner: (table_claims, text_without_tables)
        """
        table_claims: List[Dict[str, Any]] = []
        lines = text.split("\n")
        non_table_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("|") and re.match(r"^\|[\s\-:|]+\|$", next_line):
                    header_line = line
                    separator_line = next_line
                    headers = [cls.clean_markdown_inline(c.strip()) for c in header_line.split("|")[1:-1]]
                    
                    scaffold_desc = f"| {' | '.join(headers)} | karşılaştırma ve özellik tablosu iskeleti"
                    table_claims.append({
                        "claim_type": "table_scaffold",
                        "claim_text": scaffold_desc,
                        "raw_origin": header_line,
                    })

                    i += 2
                    while i < len(lines):
                        row_line = lines[i].strip()
                        if not (row_line.startswith("|") and row_line.endswith("|")):
                            break
                        cells = [cls.clean_markdown_inline(c.strip()) for c in row_line.split("|")[1:-1]]
                        if cells and any(cells):
                            row_key = cells[0]
                            if len(cells) > 1 and len(headers) >= len(cells):
                                for col_idx in range(1, len(cells)):
                                    col_name = headers[col_idx] if col_idx < len(headers) else f"Sütun {col_idx}"
                                    cell_val = cells[col_idx]
                                    if cell_val:
                                        verbalized = f"{col_name} durumunda {row_key}: {cell_val}"
                                        table_claims.append({
                                            "claim_type": "table_cell",
                                            "claim_text": verbalized,
                                            "raw_origin": row_line,
                                        })
                            else:
                                verbalized = f"Tablo verisi: {' - '.join(cells)}"
                                table_claims.append({
                                    "claim_type": "table_cell",
                                    "claim_text": verbalized,
                                    "raw_origin": row_line,
                                })
                        i += 1
                    continue
            non_table_lines.append(lines[i])
            i += 1

        remaining_text = "\n".join(non_table_lines)
        return table_claims, remaining_text

    @classmethod
    def extract_recommendations(cls, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Bakım adımlarını, öneri bloklarını ve eylem maddelerini ayıklar.
        Döner: (recommendation_claims, remaining_text)
        """
        rec_claims: List[Dict[str, Any]] = []
        lines = text.split("\n")
        non_rec_lines = []
        in_rec_section = False

        rec_section_header = re.compile(
            r"^\s*#{1,4}\s*(?:önerilen|bakım|aksiyon|müdahale|tavsiye|eylem)",
            re.IGNORECASE,
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                non_rec_lines.append(line)
                continue

            if rec_section_header.search(line_str):
                in_rec_section = True
                continue

            is_list_item = bool(re.match(r"^\s*(?:\d+[\.)]|[-*•])\s+", line_str))
            has_directive = bool(
                re.search(cls.DIRECTIVE_VERBS, line_str, re.IGNORECASE)
                or re.search(cls.MODAL_SUFFIXES, line_str, re.IGNORECASE)
            )

            if in_rec_section and is_list_item:
                cleaned_item = re.sub(r"^\s*(?:\d+[\.)]|[-*•])\s+", "", line_str)
                cleaned_item = cls.clean_markdown_inline(cleaned_item)
                if len(cleaned_item) > 10:
                    rec_claims.append({
                        "claim_type": "recommendation",
                        "claim_text": cleaned_item,
                        "raw_origin": line_str,
                    })
                continue
            elif has_directive and is_list_item:
                cleaned_item = re.sub(r"^\s*(?:\d+[\.)]|[-*•])\s+", "", line_str)
                cleaned_item = cls.clean_markdown_inline(cleaned_item)
                if len(cleaned_item) > 10:
                    rec_claims.append({
                        "claim_type": "recommendation",
                        "claim_text": cleaned_item,
                        "raw_origin": line_str,
                    })
                continue

            non_rec_lines.append(line)

        remaining_text = "\n".join(non_rec_lines)
        return rec_claims, remaining_text

    @classmethod
    def extract_narrative_sentences(cls, text: str) -> List[Dict[str, Any]]:
        """
        Kalan metni cümle bazında ayrıştırır (kısaltmaları ve float değerleri korur).
        """
        narrative_claims: List[Dict[str, Any]] = []
        raw_lines = text.split("\n")

        for line in raw_lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#") or line_str.startswith("---"):
                continue

            line_clean = re.sub(r"^\s*[-*•]\s+", "", line_str)
            cleaned = cls.clean_markdown_inline(line_clean)

            if len(cleaned) < 15 or cleaned.lower() in ("özet", "genel değerlendirme", "açıklama"):
                continue

            if not re.search(r"[.!?]$", cleaned) and len(cleaned) < 60 and not re.search(cls.DIRECTIVE_VERBS, cleaned, re.IGNORECASE):
                continue

            sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ\d])", cleaned)
            for s in sentences:
                s_clean = s.strip().rstrip(".!? ")
                if len(s_clean) >= 15:
                    narrative_claims.append({
                        "claim_type": "narrative",
                        "claim_text": s_clean,
                        "raw_origin": line_str,
                    })

        return narrative_claims

    @classmethod
    def extract_all_claims(cls, text: str) -> List[Dict[str, Any]]:
        """Metinden hiyerarşik olarak tüm iddiaları (tablo, öneri, metin) çıkarır."""
        table_claims, text_no_tables = cls.extract_tables(text)
        rec_claims, text_no_recs = cls.extract_recommendations(text_no_tables)
        narrative_claims = cls.extract_narrative_sentences(text_no_recs)

        all_claims = table_claims + narrative_claims + rec_claims
        return all_claims

class DocStoreContextLoader:
    """`docstore.db` üzerinden ID'lere göre orijinal bağlam metinlerini ve paragraflarını yükler."""

    def __init__(self, docstore_path: str = "docstore.db"):
        self.docstore_path = Path(docstore_path).resolve()

    def get_passages_for_query(
        self,
        retrieved_parent_ids: Optional[str] = None,
        reranked_chunk_ids: Optional[str] = None,
        max_paras_per_doc: int = 8,
    ) -> List[Tuple[str, str]]:
        """
        DocStore'dan çekilen dökümanları temiz paragraflara böler.
        Döner: List[(chunk_id, passage_text)]
        """
        if not self.docstore_path.exists():
            logger.warning("DocStore bulunamadı: %s", self.docstore_path)
            return []

        ids_to_fetch: List[str] = []
        if retrieved_parent_ids and str(retrieved_parent_ids) != "nan":
            ids_to_fetch.extend([x.strip() for x in str(retrieved_parent_ids).split(";") if x.strip()])
        if not ids_to_fetch and reranked_chunk_ids and str(reranked_chunk_ids) != "nan":
            ids_to_fetch.extend([x.strip() for x in str(reranked_chunk_ids).split(";") if x.strip()][:4])

        if not ids_to_fetch:
            return []

        conn = sqlite3.connect(self.docstore_path)
        cursor = conn.cursor()
        passages: List[Tuple[str, str]] = []

        for cid in ids_to_fetch:
            cursor.execute("SELECT text FROM docs WHERE id = ?;", (cid,))
            row = cursor.fetchone()
            if row and row[0]:
                raw_text = row[0]
                raw_paras = [p.strip().replace("\n", " ") for p in raw_text.split("\n\n") if len(p.strip()) >= 50]
                count = 0
                for p in raw_paras:
                    # Sayfa numaraları veya sadece resim başlığı olan satırları filtrele
                    if re.match(r"^\d+$", p) or re.match(r"^Figure\s+\d", p, re.IGNORECASE):
                        continue
                    passages.append((cid, p[:700]))
                    count += 1
                    if count >= max_paras_per_doc:
                        break

        conn.close()
        return passages


class CrossLingualNLIEvaluator:
    """
    `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` modelini kullanarak
    İngilizce Premise ve Türkçe Hypothesis çiftlerini batch olarak skorlar.
    """

    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info("NLI Evaluator yükleniyor: %s (Cihaz: %s)", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        logger.info("NLI Label Eşleştirmesi: %s", self.id2label)

    def evaluate_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[Dict[str, float]]:
        """Batch olarak Premise - Hypothesis çiftlerini değerlendirir."""
        results: List[Dict[str, float]] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            premises = [p[0] for p in chunk]
            hypotheses = [p[1] for p in chunk]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

            for p_list in probs:
                res: Dict[str, float] = {}
                for idx, p in enumerate(p_list):
                    lbl = self.id2label.get(idx, str(idx)).lower()
                    res[lbl] = float(round(p, 4))
                results.append(res)
        return results


class FaithfulnessDiagnosticAuditor:
    """
    Tüm pipeline'ı koordine eden, 13 kritik vaka üzerinde
    otonom denetim defteri (`claim_audit_ledger_df`) üreten ana sınıf.
    """

    def __init__(
        self,
        docstore_path: str = "docstore.db",
        nli_model_name: str = CrossLingualNLIEvaluator.DEFAULT_MODEL,
        cache_csv_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold_entailment: float = 0.35,
        threshold_contradiction: float = 0.50,
        threshold_ent_lower: float = 0.20,
    ):
        self.context_loader = DocStoreContextLoader(docstore_path=docstore_path)
        self.nli_model_name = nli_model_name
        self.cache_csv_path = Path(cache_csv_path) if cache_csv_path else None
        self.device = device
        self._evaluator: Optional[CrossLingualNLIEvaluator] = None

        # Kalibre edilebilir NLI eşik değerleri (varsayılanlar geriye dönük uyumludur)
        self.threshold_entailment = threshold_entailment
        self.threshold_contradiction = threshold_contradiction
        self.threshold_ent_lower = threshold_ent_lower

    @property
    def evaluator(self) -> CrossLingualNLIEvaluator:
        if self._evaluator is None:
            self._evaluator = CrossLingualNLIEvaluator(model_name=self.nli_model_name, device=self.device)
        return self._evaluator

    def classify_claim(
        self,
        claim_type: str,
        probs: Dict[str, float],
        parent_id_hint: str = "docstore",
    ) -> Tuple[str, str, str]:
        """
        NLI olasılıklarına ve iddia tipine göre (Status, Category, Provenance) atar.
        """
        p_ent = probs.get("entailment", 0.0)
        p_neu = probs.get("neutral", 0.0)
        p_con = probs.get("contradiction", 0.0)

        # 1. EŞİK: Destekli (Entailment) Kontrolü — kalibre edilebilir
        if p_ent >= self.threshold_entailment:
            status = "Destekli"
            category = "Destekli"
            provenance = f"NLI Entailment (P_ent={p_ent:.2f}, P_neu={p_neu:.2f}, P_con={p_con:.2f}) [Kaynak: {parent_id_hint}]."
            return status, category, provenance

        # 2. EŞİK: Kategori C (Gerçek Halüsinasyon / Contradiction) Kontrolü — kalibre edilebilir
        if p_ent < self.threshold_ent_lower and p_con >= self.threshold_contradiction:
            status = "Desteksiz"
            category = "Kategori C (Gerçek Halüsinasyon)"
            provenance = f"NLI Contradiction (P_con={p_con:.2f}, P_neu={p_neu:.2f}, P_ent={p_ent:.2f}) - Doğrudan teknik çelişki."
            return status, category, provenance

        status = "Desteksiz"
        if claim_type == "table_scaffold":
            category = "Kategori A (Format Artefaktı)"
            provenance = f"NLI Neutral (P_neu={p_neu:.2f}) - Sentetik Markdown tablo başlık ve iskelet formatı."
        elif claim_type == "recommendation":
            category = "Kategori B (Saha Çıkarımı)"
            provenance = f"NLI Neutral (P_neu={p_neu:.2f}, P_con={p_con:.2f}) - Genel bakım/saha müdahale önerisi, bağlamı çiğnemiyor."
        elif claim_type == "table_cell":
            category = "Kategori B (Saha Çıkarımı)"
            provenance = f"NLI Neutral (P_neu={p_neu:.2f}) - Tablo içi ek parametrik detay, bağlamla çelişmiyor."
        else:
            category = "Kategori B (Saha Çıkarımı)"
            provenance = f"NLI Neutral (P_neu={p_neu:.2f}, P_con={p_con:.2f}) - Genel mühendislik çıkarımı."

        return status, category, provenance

    def audit_cases(
        self,
        df_eval: pd.DataFrame,
        threshold: float = 0.70,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        `df_eval` içindeki `faithfulness < threshold` vakalarını (13 kritik vaka)
        otonom olarak denetler. Paragraf bazlı çapraz dilli NLI eşleştirmesi uygular.
        """
        if not force_recompute and self.cache_csv_path and self.cache_csv_path.exists():
            logger.info("Önbellekten denetim defteri yükleniyor: %s", self.cache_csv_path)
            cached_df = pd.read_csv(self.cache_csv_path)
            return cached_df

        target_df = df_eval[df_eval["faithfulness"] < threshold].copy()
        logger.info(
            "Toplam %d vaka otonom denetime alınıyor (Eşik: faithfulness < %.2f)...",
            len(target_df),
            threshold,
        )

        all_records: List[Dict[str, Any]] = []

        for _, row in target_df.iterrows():
            qid = row["_id"]
            ft = row["fault_type"]
            ans = row["generated_answer"]
            p_ids = str(row.get("retrieved_parent_ids", ""))
            c_ids = str(row.get("reranked_chunk_ids", ""))

            passages = self.context_loader.get_passages_for_query(
                retrieved_parent_ids=p_ids,
                reranked_chunk_ids=c_ids,
            )

            claims = ClaimExtractor.extract_all_claims(ans)
            if not claims or not passages:
                continue

            pairs = []
            pair_map = []
            for c_idx, c in enumerate(claims):
                for p_idx, (pid, p_text) in enumerate(passages):
                    pairs.append((p_text, c["claim_text"]))
                    pair_map.append((c_idx, p_idx))

            batch_probs = self.evaluator.evaluate_batch(pairs, batch_size=16)

            claim_scores: Dict[int, Dict[str, Any]] = {i: {"e": [], "c": [], "n": [], "best_pid": passages[0][0]} for i in range(len(claims))}
            for (c_idx, p_idx), prob in zip(pair_map, batch_probs):
                claim_scores[c_idx]["e"].append(prob.get("entailment", 0.0))
                claim_scores[c_idx]["c"].append(prob.get("contradiction", 0.0))
                claim_scores[c_idx]["n"].append(prob.get("neutral", 0.0))

            for c_idx, c_info in enumerate(claims):
                scores = claim_scores[c_idx]
                max_e = max(scores["e"]) if scores["e"] else 0.0
                max_c = max(scores["c"]) if scores["c"] else 0.0
                best_e_idx = scores["e"].index(max_e) if scores["e"] else 0
                best_pid = passages[best_e_idx][0]

                agg_probs = {
                    "entailment": max_e,
                    "contradiction": max_c,
                    "neutral": round(1.0 - max(max_e, max_c), 4),
                }

                status, category, provenance = self.classify_claim(
                    claim_type=c_info["claim_type"],
                    probs=agg_probs,
                    parent_id_hint=best_pid,
                )
                all_records.append({
                    "_id": qid,
                    "fault_type": ft,
                    "claim_type": c_info["claim_type"],
                    "claim_text": c_info["claim_text"],
                    "p_entail": max_e,
                    "p_neutral": agg_probs["neutral"],
                    "p_contra": max_c,
                    "status": status,
                    "category": category,
                    "provenance": provenance,
                })

        ledger_df = pd.DataFrame(all_records)

        # Eşik bilgilerini metadata olarak kaydet (şeffaflık / provenance)
        if len(ledger_df) > 0:
            ledger_df.attrs["threshold_entailment"] = self.threshold_entailment
            ledger_df.attrs["threshold_contradiction"] = self.threshold_contradiction
            ledger_df.attrs["threshold_ent_lower"] = self.threshold_ent_lower

        if self.cache_csv_path:
            self.cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_df.to_csv(self.cache_csv_path, index=False)
            logger.info(
                "Otonom denetim defteri kaydedildi: %s (Eşikler: θ_ent=%.2f, θ_con=%.2f, θ_ent_lower=%.2f)",
                self.cache_csv_path,
                self.threshold_entailment,
                self.threshold_contradiction,
                self.threshold_ent_lower,
            )

        return ledger_df

    @staticmethod
    def get_summary_metrics(df_claims: pd.DataFrame) -> Dict[str, Any]:
        """Ceza payları, iddia sayıları ve düzeltilmiş güvenilirlik oranlarını hesaplar."""
        total_claims = len(df_claims)
        supported_claims = (df_claims["status"] == "Destekli").sum()
        penalized_claims = (df_claims["status"] == "Desteksiz").sum()

        penalized_df = df_claims[df_claims["status"] == "Desteksiz"]
        n_pen = len(penalized_df)

        n_a = (penalized_df["category"].str.contains("Kategori A")).sum()
        n_b = (penalized_df["category"].str.contains("Kategori B")).sum()
        n_c = (penalized_df["category"].str.contains("Kategori C")).sum()

        pct_a = (n_a / n_pen * 100) if n_pen > 0 else 0.0
        pct_b = (n_b / n_pen * 100) if n_pen > 0 else 0.0
        pct_c = (n_c / n_pen * 100) if n_pen > 0 else 0.0

        adj_faithfulness = (1.0 - (n_c / total_claims)) if total_claims > 0 else 1.0

        return {
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "penalized_claims": penalized_claims,
            "n_cat_a": n_a,
            "n_cat_b": n_b,
            "n_cat_c": n_c,
            "pct_cat_a": pct_a,
            "pct_cat_b": pct_b,
            "pct_cat_c": pct_c,
            "adjusted_faithfulness": adj_faithfulness,
        }

    @staticmethod
    def render_styled_table(df_claims: pd.DataFrame, max_rows: int = 50) -> Any:
        """Notebook için görsel olarak biçimlendirilmiş Pandas Styler tablosu döner."""
        display_cols = ["_id", "fault_type", "claim_text", "status", "category", "provenance"]
        cols_to_use = [c for c in display_cols if c in df_claims.columns]
        
        sample_df = df_claims[cols_to_use].head(max_rows)
        
        def highlight_category(val):
            val_str = str(val)
            if val_str == "Destekli":
                return "background-color: #d4edda; color: #155724; font-weight: bold;"
            elif "Kategori A" in val_str:
                return "background-color: #d1ecf1; color: #0c5460;"
            elif "Kategori B" in val_str:
                return "background-color: #fff3cd; color: #856404;"
            elif "Kategori C" in val_str:
                return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
            return ""

        return (
            sample_df.style
            .map(highlight_category, subset=["category"])
            .set_caption(f"<b>Otonom Denetlenen Atomik İddia Defteri (İlk {len(sample_df)} / {len(df_claims)} İddia)</b>")
        )

    @staticmethod
    def plot_penalty_distribution(df_claims: pd.DataFrame) -> None:
        """Deterministik ceza dağılımı çubuk grafiğini çizer."""

        summary = FaithfulnessDiagnosticAuditor.get_summary_metrics(df_claims)
        n_pen = summary["penalized_claims"]
        categories = [
            f"Kategori A: Format / Tablo Artefaktı\n(N = {summary['n_cat_a']} / {n_pen})",
            f"Kategori B: Bilinçli Saha Çıkarımı\n(N = {summary['n_cat_b']} / {n_pen})",
            f"Kategori C: Gerçek Halüsinasyon\n(N = {summary['n_cat_c']} / {n_pen})",
        ]
        shares = [summary["pct_cat_a"], summary["pct_cat_b"], summary["pct_cat_c"]]
        colors = ["#1f77b4", "#ff7f0e", "#d62728"]

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
        bars = ax.bar(categories, shares, color=colors, edgecolor="black", alpha=0.88, width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"%{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_title(
            f"Otonom Ceza Ayrıştırması: {n_pen} Ceza Alan İddianın Dağılımı ({df_claims['_id'].nunique()} Vaka)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylabel("Ceza Payı Oranı (%)", fontsize=10)
        ax.set_ylim(0, max(max(shares) + 15, 80))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
