# 📊 VibraDiag Evaluation Report — `stage2_hybrid_corrector`

![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square&logo=github)
![Questions](https://img.shields.io/badge/Dataset-8%20Queries-blue?style=flat-square)
![Evaluator](https://img.shields.io/badge/Evaluator-RAGAS%20%2B%20Deterministic-orange?style=flat-square)

- 📅 **Çalıştırma zamanı:** `2026-09-01T12:55:26`
- 🎯 **Gold dataset:** `data/evaluation/retrieval_benchmark_dev8.json` (8 soru)

## ⚙️ System & Pipeline Configuration

| Kategori | Parametre | Değer |
| :--- | :--- | :--- |
| **Dataset** | `gold_dataset` | `data/evaluation/retrieval_benchmark_dev8.json` |
| **Dataset** | `n_questions` | `8` |
| **Reranker** | `text_top_k (final, reranker sonrası)` | `5` |
| **Retrieval** | `visual_top_k` | `3` |
| **Retrieval** | `deduplicate_parents (eval)` | `False` |
| **Retrieval** | `enable_corrector (eval)` | `True` |
| **Retrieval** | `visual_fallback_threshold (kullanılan)` | `0.2 [retrieval.yaml]` |
| **Retrieval** | `config_source` | `retrieval.yaml + app.yaml` |
| **Retrieval** | `decomposer_enabled` | `False` |
| **Embeddings** | `embedder_model` | `BAAI/bge-m3` |
| **Embeddings** | `embedder_device` | `mps` |
| **Embeddings** | `sparse_encoder_model` | `SparseTextEmbedding` |
| **Reranker** | `reranker_model` | `BAAI/bge-reranker-large` |
| **Reranker** | `reranker_device` | `mps` |
| **Reranker** | `reranker_candidate_pool_rerank_top_k` | `25` |
| **Reranker** | `reranker_score_threshold` | `0.2` |
| **Reranker** | `reranker_soft_fallback_floor` | `0.1` |
| **Retrieval** | `visual_fallback_threshold` | `0.2` |
| **RAGAS Judge** | `ragas_judge_model` | `gemini-3.1-flash-lite` |
| **RAGAS Judge** | `ragas_metrics` | `faithfulness, answer_relevancy, llm_context_precision_with_reference, context_recall` |
| **Generation** | `generator_enabled` | `True` |
| **Generation** | `generator_model` | `openai/gpt-oss-120b` |
| **Generation** | `generator_temperature` | `0.2` |
| **Generation** | `generator_max_tokens` | `4096` |
| **Retrieval** | `retrieval_concurrency` | `4` |
| **RAGAS Judge** | `judge_concurrency` | `2` |

## 🎯 Deterministik ID-Seviyesi Metrikler (LLM Bağımsız)

| Metrik Kümeleri | Metrik | Skor | Derece |
| :--- | :--- | :---: | :---: |
| **Child Chunk** | `det_child_recall` | **0.6458** | 🟡 ⭐⭐⭐⭐☆ |
| **Child Chunk** | `det_child_precision` | **0.3000** | 🔴 ⭐⭐☆☆☆ |
| **Child Chunk** | `det_child_hit_rate` | **0.7500** | 🟡 ⭐⭐⭐⭐☆ |
| **Child Chunk** | `det_child_mrr` | **0.5417** | 🟠 ⭐⭐⭐☆☆ |
| **Child Chunk** | `det_child_ndcg` | **0.5109** | 🟠 ⭐⭐⭐☆☆ |
| **Parent Chunk** | `det_parent_recall` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_precision` | **0.4792** | 🟠 ⭐⭐⭐☆☆ |
| **Parent Chunk** | `det_parent_hit_rate` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_mrr` | **0.7917** | 🟡 ⭐⭐⭐⭐☆ |
| **Parent Chunk** | `det_parent_ndcg` | **0.8452** | 🟢 ⭐⭐⭐⭐⭐ |
| **Visual Chunk** | `det_visual_recall` | **0.0000** | ⚪ ☆☆☆☆☆ |

## ✨ Aggregate RAGAS Metrikleri

| Metrik Kategorisi | RAGAS Metrik | Skor | Derece |
| :--- | :--- | :---: | :---: |
| **Generation** | `faithfulness` | **0.7780** | 🟡 ⭐⭐⭐⭐☆ |
| **Generation** | `answer_relevancy` | **0.8303** | 🟢 ⭐⭐⭐⭐⭐ |
| **Semantic Retrieval** | `llm_context_precision_with_reference` | **0.7547** | 🟡 ⭐⭐⭐⭐☆ |
| **Semantic Retrieval** | `context_recall` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |

## 🔍 Fault Type Kırılımı

| Fault Type | `answer_relevancy` | `context_recall` | `det_child_hit_rate` | `det_child_mrr` | `det_child_ndcg` | `det_child_precision` | `det_child_recall` | `det_parent_hit_rate` | `det_parent_mrr` | `det_parent_ndcg` | `det_parent_precision` | `det_parent_recall` | `det_visual_recall` | `faithfulness` | `llm_context_precision_with_reference` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **standards** | **0.8651** | **1.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **0.5000** | **1.0000** | **0.0000** | **0.9412** | **1.0000** |
| **signal_processing** | **0.8577** | **1.0000** | **1.0000** | **0.5000** | **0.3869** | **0.2000** | **0.5000** | **1.0000** | **0.5000** | **0.6309** | **0.3333** | **1.0000** | **0.0000** | **0.5714** | **0.5889** |
| **unbalance** | **0.8932** | **1.0000** | **1.0000** | **0.5000** | **0.4776** | **0.4000** | **0.6667** | **1.0000** | **0.5000** | **0.6309** | **0.5000** | **1.0000** | **0.0000** | **0.8235** | **0.5889** |
| **misalignment** | **0.8807** | **1.0000** | **1.0000** | **0.3333** | **0.6183** | **0.6000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | **0.6667** | **1.0000** |
| **looseness** | **0.7689** | **1.0000** | **1.0000** | **1.0000** | **0.8772** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.8824** | **0.7500** |
| **resonance** | **0.7702** | **1.0000** | **1.0000** | **1.0000** | **0.8503** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.5000** | **1.0000** | **0.0000** | **0.9412** | **0.8875** |
| **flow_hydrodynamic** | **0.8744** | **1.0000** | **1.0000** | **1.0000** | **0.8772** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.6667** | **0.8056** |
| **bearing_fault** | **0.7320** | **1.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **1.0000** | **0.3333** | **0.5000** | **0.3333** | **1.0000** | **0.0000** | **0.7308** | **0.4167** |

## 📋 Soru Bazlı Detaylı Kıyaslama Tablosu

| ID | Soru | Kategori | Hit (Parent) | Beklenen Parent | Çekilen Parent | Context Durumu |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| `q_005` | ISO 10816 standartları ve titreşim şiddet tab... | `standards` | **✅ Başarılı** | `text_parent_15` | `text_parent_15, text_parent_17` | 🟢 5 chunk |
| `q_009` | FFT analizinde pencereleme (Windowing / Leaka... | `signal_processing` | **✅ Başarılı** | `text_parent_21` | `text_parent_19, text_parent_21, text_parent_22` | 🟢 5 chunk |
| `q_014` | Rotorda dengesizlik (Unbalance) arızası spekt... | `unbalance` | **✅ Başarılı** | `text_parent_28` | `text_parent_34, text_parent_28` | 🟢 5 chunk |
| `q_016` | Açısal eksen kaçıklığı (Angular Misalignment)... | `misalignment` | **✅ Başarılı** | `text_parent_31` | `text_parent_31` | 🟢 5 chunk |
| `q_017` | Mekanik gevşeklik (Mechanical Looseness) arız... | `looseness` | **✅ Başarılı** | `text_parent_32` | `text_parent_32, text_parent_38, text_parent_31` | 🟢 5 chunk |
| `q_018` | Rezonans (Resonance) ve doğal frekans tespiti... | `resonance` | **✅ Başarılı** | `text_parent_33` | `text_parent_33, text_parent_46` | 🟢 5 chunk |
| `q_019` | Pompalarda kanat geçiş frekansı (Vane Pass Fr... | `flow_hydrodynamic` | **✅ Başarılı** | `text_parent_36` | `text_parent_36, text_parent_37, text_parent_33` | 🟢 5 chunk |
| `q_020` | Rulman arızalarının gelişme aşamalarında (Sta... | `bearing_fault` | **✅ Başarılı** | `text_parent_39` | `text_parent_27, text_parent_31, text_parent_39` | 🟢 5 chunk |

---

## 📌 Proje Geliştiricisi Değerlendirme Notu

> [!NOTE]
> **VibraDiag MVP Kapsam ve Metodoloji Notu:**  
> *VibraDiag MVP düzeyinde, corpus'un özellikle mekanik arızalara cevaben sahada yapılması gereken operasyonel müdahale ve koruyucu bakım adımları bağlamındaki kısıtlılığından ötürü, prompt stratejisi sistemin yalnızca teorik/sığ tanımlar yerine sahada uygulanabilir, kapsamlı ve teknik bir kılavuz sunabilmesi amacıyla **'kontrollü parametrik alan çıkarımı'** yapacak biçimde yeniden yapılandırılmıştır.*  
> 
> *Bu yaklaşım klasik saf-RAG sadakati açısından bir trade-off barındırmaktadır; ancak teşhis çekirdeği (arıza tipi, FFT frekans harmonikleri, kinematik rulman frekansları) dokümana ve deterministik sinyal motoruna **%100 sadık** tutulurken, bakım aksiyonları ise aşırı halüsinasyonları ve spekülatif sayısal toleransları engelleyecek net sınırlarla zenginleştirilmiştir. Çıktılar manuel olarak doğrulanmış, sıfır TPM/413 hatası ile son derece tatmin edici ve minimal halüsinasyon riskine sahip bir dengeye ulaştırılmıştır. Bu tercih saf bir best-practice iddiası taşımamakta olup, MVP seviyesinde pratik endüstriyel fayda üretmek amacıyla bilinçli olarak seçilmiştir.*

