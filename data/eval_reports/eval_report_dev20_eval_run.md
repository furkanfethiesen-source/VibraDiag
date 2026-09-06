# 📊 VibraDiag Evaluation Report — `dev20_eval_run`

![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square&logo=github)
![Questions](https://img.shields.io/badge/Dataset-20%20Queries-blue?style=flat-square)
![Evaluator](https://img.shields.io/badge/Evaluator-RAGAS%20%2B%20Deterministic-orange?style=flat-square)

- 📅 **Çalıştırma zamanı:** `2026-09-03T08:19:58`
- 🎯 **Gold dataset:** `data/evaluation/retrieval_benchmark_dev20.json` (20 soru)

## ⚙️ System & Pipeline Configuration

| Kategori | Parametre | Değer |
| :--- | :--- | :--- |
| **Dataset** | `gold_dataset` | `data/evaluation/retrieval_benchmark_dev20.json` |
| **Dataset** | `n_questions` | `20` |
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
| **Generation** | `generator_max_tokens` | `1500` |
| **Retrieval** | `retrieval_concurrency` | `4` |
| **RAGAS Judge** | `judge_concurrency` | `2` |

## 🎯 Deterministik ID-Seviyesi Metrikler (LLM Bağımsız)

| Metrik Kümeleri | Metrik | Skor | Derece |
| :--- | :--- | :---: | :---: |
| **Child Chunk** | `det_child_recall` | **0.8617** | 🟢 ⭐⭐⭐⭐⭐ |
| **Child Chunk** | `det_child_precision` | **0.3700** | 🔴 ⭐⭐☆☆☆ |
| **Child Chunk** | `det_child_hit_rate` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |
| **Child Chunk** | `det_child_mrr` | **0.8117** | 🟢 ⭐⭐⭐⭐⭐ |
| **Child Chunk** | `det_child_ndcg` | **0.7697** | 🟡 ⭐⭐⭐⭐☆ |
| **Parent Chunk** | `det_parent_recall` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_precision` | **0.4000** | 🟠 ⭐⭐⭐☆☆ |
| **Parent Chunk** | `det_parent_hit_rate` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_mrr` | **0.8208** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_ndcg` | **0.8662** | 🟢 ⭐⭐⭐⭐⭐ |
| **Visual Chunk** | `det_visual_recall` | **0.0000** | ⚪ ☆☆☆☆☆ |

## ✨ Aggregate RAGAS Metrikleri

| Metrik Kategorisi | RAGAS Metrik | Skor | Derece |
| :--- | :--- | :---: | :---: |
| **Generation** | `faithfulness` | **0.6262** | 🟡 ⭐⭐⭐⭐☆ |
| **Generation** | `answer_relevancy` | **0.8564** | 🟢 ⭐⭐⭐⭐⭐ |
| **Semantic Retrieval** | `llm_context_precision_with_reference` | **0.7967** | 🟡 ⭐⭐⭐⭐☆ |
| **Semantic Retrieval** | `context_recall` | **0.9750** | 🟢 ⭐⭐⭐⭐⭐ |

## 🔍 Fault Type Kırılımı

| Fault Type | `answer_relevancy` | `context_recall` | `det_child_hit_rate` | `det_child_mrr` | `det_child_ndcg` | `det_child_precision` | `det_child_recall` | `det_parent_hit_rate` | `det_parent_mrr` | `det_parent_ndcg` | `det_parent_precision` | `det_parent_recall` | `det_visual_recall` | `faithfulness` | `llm_context_precision_with_reference` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **standards** | **0.9449** | **1.0000** | **1.0000** | **1.0000** | **0.7039** | **0.4000** | **0.6667** | **1.0000** | **1.0000** | **1.0000** | **0.5000** | **1.0000** | **0.0000** | **0.5625** | **1.0000** |
| **signal_processing** | **0.9102** | **1.0000** | **1.0000** | **0.5500** | **0.6021** | **0.3500** | **0.7917** | **1.0000** | **0.5833** | **0.6905** | **0.3333** | **1.0000** | **0.0000** | **0.6607** | **0.5944** |
| **unbalance** | **0.7834** | **1.0000** | **1.0000** | **0.5000** | **0.3869** | **0.2000** | **0.5000** | **1.0000** | **0.5000** | **0.6309** | **0.5000** | **1.0000** | **0.0000** | **0.9048** | **0.5889** |
| **misalignment** | **0.7283** | **1.0000** | **1.0000** | **1.0000** | **0.9060** | **0.6000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | **0.6765** | **1.0000** |
| **looseness** | **0.7973** | **1.0000** | **1.0000** | **1.0000** | **0.8772** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.5200** | **0.7500** |
| **resonance** | **0.8413** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.5000** | **1.0000** | **0.0000** | **0.4286** | **0.8875** |
| **flow_hydrodynamic** | **0.7205** | **1.0000** | **1.0000** | **1.0000** | **0.8772** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.4000** | **0.8056** |
| **bearing_fault** | **0.8644** | **1.0000** | **1.0000** | **0.6667** | **0.6578** | **0.4000** | **0.7000** | **1.0000** | **0.6667** | **0.7500** | **0.3333** | **1.0000** | **0.0000** | **0.8117** | **0.7083** |
| **monitoring** | **0.8670** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.2500** | **1.0000** | **0.0000** | **0.1111** | **1.0000** |
| **fundamentals** | **0.9354** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.9375** | **1.0000** |
| **eccentricity** | **0.8691** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.5217** | **1.0000** |
| **bent_shaft** | **0.9641** | **0.5000** | **1.0000** | **0.2000** | **0.3869** | **0.2000** | **1.0000** | **1.0000** | **0.2500** | **0.4307** | **0.2500** | **1.0000** | **0.0000** | **0.3500** | **0.3250** |
| **electrical** | **0.7893** | **1.0000** | **1.0000** | **1.0000** | **0.9197** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.4762** | **0.8333** |
| **aerodynamic** | **0.8342** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.3333** | **1.0000** | **0.0000** | **0.6842** | **1.0000** |
| **journal_bearing** | **0.9400** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.5000** | **1.0000** | **0.0000** | **0.6842** | **1.0000** |
| **gear_fault** | **0.7436** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.4000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.5000** | **1.0000** | **0.0000** | **1.0000** | **0.9500** |

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
| `q_021` | Bir titreşim analiz sisteminin temel bileşenl... | `monitoring` | **✅ Başarılı** | `text_parent_7` | `text_parent_7, text_parent_42, text_parent_22, text_parent_5` | 🟢 5 chunk |
| `q_022` | Titreşim hareketinde bir çevrim (one cycle of... | `fundamentals` | **✅ Başarılı** | `text_parent_12` | `text_parent_12, text_parent_47, text_parent_26` | 🟢 5 chunk |
| `q_023` | Analog ve dijital sinyal (Analog/Digital Sign... | `signal_processing` | **✅ Başarılı** | `text_parent_16` | `text_parent_16, text_parent_19, text_parent_17` | 🟢 5 chunk |
| `q_024` | FFT analizinde Örnek Sayısı (N), Kayıt Süresi... | `signal_processing` | **✅ Başarılı** | `text_parent_20` | `text_parent_19, text_parent_22, text_parent_20` | 🟢 5 chunk |
| `q_025` | Cepstrum analizi nedir ve hangi tür arızaları... | `signal_processing` | **✅ Başarılı** | `text_parent_26` | `text_parent_41, text_parent_26, text_parent_25` | 🟢 5 chunk |
| `q_026` | Eksantriklik (Eccentricity) arızası faz ölçüm... | `eccentricity` | **✅ Başarılı** | `text_parent_29` | `text_parent_29, text_parent_34, text_parent_28` | 🟢 5 chunk |
| `q_027` | Eğik şaft (Bent Shaft) arızasında eksenel tit... | `bent_shaft` | **✅ Başarılı** | `text_parent_30` | `text_parent_31, text_parent_29, text_parent_32, text_parent_30` | 🟢 5 chunk |
| `q_028` | Titreşim spektrumunda elektriksel kaynaklı bi... | `electrical` | **✅ Başarılı** | `text_parent_34` | `text_parent_34, text_parent_27, text_parent_41` | 🟢 5 chunk |
| `q_029` | Fanlarda ve blowerlarda akış türbülansı (flow... | `aerodynamic` | **✅ Başarılı** | `text_parent_37` | `text_parent_37, text_parent_49, text_parent_36` | 🟢 5 chunk |
| `q_030` | Yağ girdabı (Oil Whirl) arızası hangi frekans... | `journal_bearing` | **✅ Başarılı** | `text_parent_38` | `text_parent_38, text_parent_39` | 🟢 5 chunk |
| `q_031` | Rulman karakteristik arıza frekansları (BPFO,... | `bearing_fault` | **✅ Başarılı** | `text_parent_39` | `text_parent_39, text_parent_18, text_parent_34` | 🟢 5 chunk |
| `q_032` | Dişli kutusu (gearbox) titreşim spektrumunda ... | `gear_fault` | **✅ Başarılı** | `text_parent_41` | `text_parent_41, text_parent_49` | 🟢 5 chunk |
