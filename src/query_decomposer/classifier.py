import os
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from loguru import logger

from schemas.schemas import ClassifierResult
from config_loader import load_appcfg

TR_ASCII_TRANS = str.maketrans("çğıöşü", "cgiosu")


def normalize_turkish(text: str) -> str:
    """Türkçe ve özel karakterleri normalize edip ASCII karşılıklarına dönüştürür."""
    if not text:
        return ""
    text_lower = text.replace("İ", "i").replace("I", "ı").lower()
    return text_lower.translate(TR_ASCII_TRANS)

class LexicalFeatures(BaseEstimator, TransformerMixin):
    """Sorgu metninden sayısal ve yapısal feature'lar çıkaran transformer.
    
    TF-IDF n-gram feature'larını tamamlayarak modelin kısa-telegrafik
    complex ve uzun-detaylı simple sorguları daha tutarlı ayırt etmesini sağlar.
    """

    _RE_MI_MI = re.compile(r'\w+\s+m[iı]\s+\w+\s+m[iı]', re.IGNORECASE)
    _RE_VS = re.compile(r'\bvs\b')
    _RE_COMPARISON = re.compile(r'\bfark\b|\bayrim\b|\bayirt\b|\bkiyasla\b|\bkarsilastir')
    _RE_ALTERNATIVE = re.compile(r'\byoksa\b|\bya da\b|\bveya\b')
    _RE_QUESTION_SUFFIX = re.compile(r'\b(nedir|nasil|nelerdir|nasildir)\b')
    
    def __init__(self, domain_terms: set[str] | None = None):
        self.domain_terms = domain_terms or set()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._featurize(q) for q in X], dtype=np.float64)
    
    def _featurize(self, query: str) -> list[float]:
        tokens = query.split()
        n_tokens = len(tokens)
        unique_tokens = set(tokens)
        
        return [
            float(n_tokens),                                         
            float(len(query)),                                        
            float(bool(self._RE_MI_MI.search(query))),                
            float(bool(self._RE_VS.search(query))),                     
            float(bool(self._RE_COMPARISON.search(query))),           
            float(bool(self._RE_ALTERNATIVE.search(query))),          
            float(len(unique_tokens & self.domain_terms)),            
            float(bool(self._RE_QUESTION_SUFFIX.search(query))),      
            len(unique_tokens) / n_tokens if n_tokens > 0 else 0.0,  
        ]

class QueryComplexityClassifier:
    """Sorgu karmaşıklığını tahmin eden LR tabanlı sınıflandırıcı."""
    
    def __init__(self, model_path: str | None = None):
        app_cfg = load_appcfg()
        decomposer_cfg = getattr(app_cfg, "decomposer", {}) or {}
        self.confidence_threshold = decomposer_cfg.get("confidence_threshold", 0.55)
        
        if model_path is None:
            model_path = decomposer_cfg.get("model_path", "src/query_decomposer/models/complexity_pipeline.pkl")
            
        abs_model_path = Path(model_path).resolve()

        if not abs_model_path.exists():
            raise FileNotFoundError(
                f"Complexity classifier modeli bulunamadı: {abs_model_path}\n"
                f"Model dosyası build aşamasında oluşturulmalıdır.\n"
                f"Çalıştırın: uv run python src/query_decomposer/train_classifier.py"
            )
            
        logger.info(f"Karmaşıklık modeli yükleniyor: {abs_model_path}")
        self.pipeline = joblib.load(abs_model_path)

    def predict(self, query: str) -> ClassifierResult:
        """Sorguyu normalize edip karmaşıklığını tahmin eder."""
        norm_query = normalize_turkish(query)
        probs = self.pipeline.predict_proba([norm_query])[0]

        complex_prob = float(probs[1]) if len(probs) > 1 else 0.0
        is_complex = complex_prob >= self.confidence_threshold
        
        confidence = complex_prob if is_complex else 1.0 - complex_prob
        
        return ClassifierResult(
            is_complex=is_complex,
            confidence=confidence
        )

    @classmethod
    def train_and_save(cls, jsonl_path: str, output_path: str) -> dict:
        """Verilen JSONL dosyasıyla modeli eğitir, kaydeder ve metrikleri döndürür."""
        logger.info(f"Eğitim verisi okunuyor: {jsonl_path}")
        queries = []
        labels = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                q = normalize_turkish(data.get("query", ""))
                label = 1 if data.get("label", "simple") == "complex" else 0
                queries.append(q)
                labels.append(label)
        
        X = queries
        y = labels

        domain_terms_path = Path(jsonl_path).parent / "domain_terms.json"
        if domain_terms_path.exists():
            with open(domain_terms_path, "r", encoding="utf-8") as dt_f:
                domain_terms: set[str] = set(json.load(dt_f).get("terms", []))
            logger.info(f"{len(domain_terms)} domain terimi yüklendi: {domain_terms_path}")
        else:
            logger.warning(
                f"Domain terim dosyası bulunamadı: {domain_terms_path}. "
                "Chi2 analizi ile otomatik oluşturuluyor..."
            )
            from query_decomposer.build_domain_terms import extract_and_save_domain_terms
            domain_terms = extract_and_save_domain_terms(
                data_path=jsonl_path,
                output_path=domain_terms_path,
            )
            logger.info(
                f"Otomatik oluşturuldu: {len(domain_terms)} terim → {domain_terms_path}"
            )

        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('word_tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=3)),
                ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, min_df=3)),
                ('lexical_scaled', Pipeline([
                    ('extract', LexicalFeatures(domain_terms=domain_terms)),
                    ('scale', StandardScaler()),
                ])),
            ])),
            ('clf', LogisticRegression(C=10.0, max_iter=1000, random_state=42))
        ])
        
        logger.info("5-fold Stratified CV ile model değerlendiriliyor...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X, y, cv=cv,
            scoring=('accuracy', 'f1', 'roc_auc', 'precision', 'recall')
        )
        
        metrics = {
            "accuracy": float(np.mean(cv_results['test_accuracy'])),
            "accuracy_std": float(np.std(cv_results['test_accuracy'])),
            "f1": float(np.mean(cv_results['test_f1'])),
            "f1_std": float(np.std(cv_results['test_f1'])),
            "precision": float(np.mean(cv_results['test_precision'])),
            "precision_std": float(np.std(cv_results['test_precision'])),
            "recall": float(np.mean(cv_results['test_recall'])),
            "recall_std": float(np.std(cv_results['test_recall'])),
            "roc_auc": float(np.mean(cv_results['test_roc_auc'])),
            "roc_auc_std": float(np.std(cv_results['test_roc_auc'])),
            "total_samples": len(X),
            "complex_samples": int(sum(y)),
            "simple_samples": int(len(y) - sum(y)),
        }
        
        logger.info("Model tüm veri seti üzerinde eğitiliyor...")
        pipeline.fit(X, y)
        
        out_dir = os.path.dirname(output_path)
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(pipeline, output_path)
        logger.info(f"Model başarıyla kaydedildi: {output_path}")
        
        return metrics
