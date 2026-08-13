# Logistic Regression Query Complexity Classifier Raporu

- **Oluşturulma Tarihi:** `2026-08-13 11:50:41`
- **Eğitim Veri Yolu:** `data/decomposer_data/training.jsonl`
- **Model Çıktı Yolu:** `src/query_decomposer/models/complexity_pipeline.pkl`
- **Toplam Örnek Sayısı:** 808 (Complex: 328, Simple: 480)

## 1. En İyi Hiperparametreler (Best Hyperparameters)

| Parameter / Component | Value | Description |
| :--- | :--- | :--- |
| **Classifier** | `LogisticRegression` | Model / Pipeline Konfigürasyonu |
| **C (Inverse Regularization)** | `10.0` | Model / Pipeline Konfigürasyonu |
| **max_iter** | `1000` | Model / Pipeline Konfigürasyonu |
| **random_state** | `42` | Model / Pipeline Konfigürasyonu |
| **Word TF-IDF n-gram** | `(1, 3)` | Model / Pipeline Konfigürasyonu |
| **Word TF-IDF sublinear_tf** | `True` | Model / Pipeline Konfigürasyonu |
| **Word TF-IDF min_df** | `3` | Model / Pipeline Konfigürasyonu |
| **Char TF-IDF analyzer** | `char_wb` | Model / Pipeline Konfigürasyonu |
| **Char TF-IDF n-gram** | `(2, 5)` | Model / Pipeline Konfigürasyonu |
| **Char TF-IDF min_df** | `3` | Model / Pipeline Konfigürasyonu |
| **Lexical Features** | `Enabled (StandardScaler)` | Model / Pipeline Konfigürasyonu |
| **Cross-Validation** | `5-Fold StratifiedKFold (shuffle=True, random_state=42)` | Model / Pipeline Konfigürasyonu |

## 2. Cross-Validation & Değerlendirme Sonuçları (5-Fold Stratified CV)

| Metrik | Ortalama (Mean) | Standart Sapma (Std) |
| :--- | :---: | :---: |
| **Accuracy** | 0.9604 | ±0.0114 |
| **F1-Score** | 0.9493 | ±0.0155 |
| **Precision** | 0.9842 | ±0.0166 |
| **Recall** | 0.9178 | ±0.0340 |
| **ROC-AUC** | 0.9893 | ±0.0127 |

## 3. Özellik Çıkarımı (Feature Union Architecture)

1. **Word TF-IDF Vectorizer**: Sorgudaki kelime n-gram ilişkilerini modeller (1-3 n-gram).
2. **Char TF-IDF Vectorizer**: Kelime sınırları içerisindeki sub-word/karakter n-gram desenlerini yakalar (2-5 char_wb).
3. **Lexical Features & StandardScaler**: Sorgu uzunluğu, domain terimleri sıklığı, karşılaştırma ve soru yapılarını sayısal vektöre dönüştürüp ölçeklendirir.
