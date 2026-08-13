#!/usr/bin/env python3
"""Standalone LR classifier eğitim script'i.

Kullanım:
    uv run python src/query_decomposer/train_classifier.py
    uv run python src/query_decomposer/train_classifier.py --data data/decomposer_data/training.jsonl --output src/query_decomposer/models/complexity_pipeline.pkl
    uv run python src/query_decomposer/train_classifier.py --compare
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query_decomposer.classifier import QueryComplexityClassifier, normalize_turkish, LexicalFeatures


def _read_data(data_path: str) -> tuple[list[str], list[int]]:
    """Eğitim verisini oku ve normalize et."""
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(normalize_turkish(item.get("query", "")))
            labels.append(1 if item.get("label", "simple") == "complex" else 0)
    return texts, labels


def _load_domain_terms(data_path: str) -> set[str]:
    """Domain terimlerini JSON'dan yükle."""
    domain_terms_path = Path(data_path).parent / "domain_terms.json"
    if domain_terms_path.exists():
        with open(domain_terms_path, "r", encoding="utf-8") as f:
            terms = set(json.load(f).get("terms", []))
        print(f"  {len(terms)} domain terimi yüklendi: {domain_terms_path}")
        return terms
    print(f"  Domain terim dosyası bulunamadı: {domain_terms_path}. Boş set kullanılıyor.")
    return set()


def run_comparison(data_path: str) -> None:
    """Üç pipeline konfigürasyonunu 5-fold Stratified CV ile karşılaştırır."""
    texts, labels = _read_data(data_path)
    domain_terms = _load_domain_terms(data_path)

    print(f"\n  Veri: {len(texts)} sorgu ({sum(labels)} complex, {len(texts) - sum(labels)} simple)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ("accuracy", "f1", "precision", "recall", "roc_auc")

    pipelines = {
        "baseline": Pipeline([
            ("features", FeatureUnion([
                ("word_tfidf", TfidfVectorizer(
                    ngram_range=(1, 3), sublinear_tf=True, min_df=3
                )),
                ("char_tfidf", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3
                )),
            ])),
            ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
        ]),
        "+lexical": Pipeline([
            ("features", FeatureUnion([
                ("word_tfidf", TfidfVectorizer(
                    ngram_range=(1, 3), sublinear_tf=True, min_df=3
                )),
                ("char_tfidf", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3
                )),
                ("lexical_scaled", Pipeline([
                    ("extract", LexicalFeatures(domain_terms=domain_terms)),
                    ("scale", StandardScaler()),
                ])),
            ])),
            ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
        ]),
        "+lexical(no_scale)": Pipeline([
            ("features", FeatureUnion([
                ("word_tfidf", TfidfVectorizer(
                    ngram_range=(1, 3), sublinear_tf=True, min_df=3
                )),
                ("char_tfidf", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3
                )),
                ("lexical", LexicalFeatures(domain_terms=domain_terms)),
            ])),
            ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
        ]),
    }

    print("\n" + "=" * 100)
    print("PIPELINE KARŞILAŞTIRMASI (5-Fold Stratified CV)")
    print("=" * 100)
    header = f"{'Pipeline':<22}" + "".join(f"{m:<18}" for m in scoring)
    print(header)
    print("-" * 100)

    for name, p in pipelines.items():
        results = cross_validate(p, texts, labels, cv=cv, scoring=scoring)

        row = f"{name:<22}"
        for metric in scoring:
            vals = results[f"test_{metric}"]
            row += f"{np.mean(vals):.4f}±{np.std(vals):.4f}  "
        print(row)

    print("=" * 100)


from datetime import datetime


def export_markdown_report(
    metrics: dict,
    data_path: str,
    output_path: str,
    report_path: str = "data/decomposer_data/lr_report.md",
    hyperparameters: dict | None = None,
) -> Path:
    """Eğitim sonuçlarını, en iyi hiperparametreleri ve CV metriklerini Markdown tablosu olarak kaydeder."""
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    if hyperparameters is None:
        hyperparameters = {
            "Classifier": "LogisticRegression",
            "C (Inverse Regularization)": 10.0,
            "max_iter": 1000,
            "random_state": 42,
            "Word TF-IDF n-gram": "(1, 3)",
            "Word TF-IDF sublinear_tf": True,
            "Word TF-IDF min_df": 3,
            "Char TF-IDF analyzer": "char_wb",
            "Char TF-IDF n-gram": "(2, 5)",
            "Char TF-IDF min_df": 3,
            "Lexical Features": "Enabled (StandardScaler)",
            "Cross-Validation": "5-Fold StratifiedKFold (shuffle=True, random_state=42)",
        }

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total = metrics.get("total_samples", "N/A")
    complex_cnt = metrics.get("complex_samples", "N/A")
    simple_cnt = metrics.get("simple_samples", "N/A")

    content = [
        "# Logistic Regression Query Complexity Classifier Raporu",
        "",
        f"- **Oluşturulma Tarihi:** `{now_str}`",
        f"- **Eğitim Veri Yolu:** `{data_path}`",
        f"- **Model Çıktı Yolu:** `{output_path}`",
        f"- **Toplam Örnek Sayısı:** {total} (Complex: {complex_cnt}, Simple: {simple_cnt})",
        "",
        "## 1. En İyi Hiperparametreler (Best Hyperparameters)",
        "",
        "| Parameter / Component | Value | Description |",
        "| :--- | :--- | :--- |",
    ]

    for param, val in hyperparameters.items():
        content.append(f"| **{param}** | `{val}` | Model / Pipeline Konfigürasyonu |")

    content.extend([
        "",
        "## 2. Cross-Validation & Değerlendirme Sonuçları (5-Fold Stratified CV)",
        "",
        "| Metrik | Ortalama (Mean) | Standart Sapma (Std) |",
        "| :--- | :---: | :---: |",
    ])

    metric_names = [
        ("Accuracy", "accuracy", "accuracy_std"),
        ("F1-Score", "f1", "f1_std"),
        ("Precision", "precision", "precision_std"),
        ("Recall", "recall", "recall_std"),
        ("ROC-AUC", "roc_auc", "roc_auc_std"),
    ]

    for label, mean_key, std_key in metric_names:
        mean_val = metrics.get(mean_key)
        std_val = metrics.get(std_key)
        if isinstance(mean_val, float):
            mean_str = f"{mean_val:.4f}"
            std_str = f"±{std_val:.4f}" if isinstance(std_val, float) else "N/A"
            content.append(f"| **{label}** | {mean_str} | {std_str} |")

    content.extend([
        "",
        "## 3. Özellik Çıkarımı (Feature Union Architecture)",
        "",
        "1. **Word TF-IDF Vectorizer**: Sorgudaki kelime n-gram ilişkilerini modeller (1-3 n-gram).",
        "2. **Char TF-IDF Vectorizer**: Kelime sınırları içerisindeki sub-word/karakter n-gram desenlerini yakalar (2-5 char_wb).",
        "3. **Lexical Features & StandardScaler**: Sorgu uzunluğu, domain terimleri sıklığı, karşılaştırma ve soru yapılarını sayısal vektöre dönüştürüp ölçeklendirir.",
        "",
    ])

    report_text = "\n".join(content)
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\n[INFO] Markdown raporu kaydedildi: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description="Query complexity LR classifier eğitimi")
    parser.add_argument(
        "--data",
        default="data/decomposer_data/training.jsonl",
        help="Eğitim verisi JSONL dosya yolu",
    )
    parser.add_argument(
        "--output",
        default="src/query_decomposer/models/complexity_pipeline.pkl",
        help="Eğitilmiş model çıktı yolu",
    )
    parser.add_argument(
        "--report",
        default="data/decomposer_data/lr_report.md",
        help="Markdown rapor çıktı yolu",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Pipeline karşılaştırma deneyini çalıştır (model kaydetmez)",
    )
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.data)
        return

    print(f"Eğitim verisi: {args.data}")
    print(f"Model çıktı: {args.output}")
    print("-" * 60)

    metrics = QueryComplexityClassifier.train_and_save(
        jsonl_path=args.data,
        output_path=args.output,
    )

    print("\n" + "=" * 60)
    print("EĞİTİM SONUÇLARI (5-Fold Stratified CV)")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nModel kaydedildi: {args.output}")

    # Markdown raporunu kaydet
    export_markdown_report(
        metrics=metrics,
        data_path=args.data,
        output_path=args.output,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()

