# 📊 VibraDiag Evaluation Report — `dev20_strict_prompt`

![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square&logo=github)
![Questions](https://img.shields.io/badge/Dataset-20%20Queries-blue?style=flat-square)
![Evaluator](https://img.shields.io/badge/Evaluator-RAGAS%20%2B%20Deterministic-orange?style=flat-square)

- 📅 **Çalıştırma zamanı:** `2026-09-03T12:03:15`
- 🎯 **Gold dataset:** `./data/evaluation/retrieval_benchmark_dev20.json` (20 soru)

## ⚙️ System & Pipeline Configuration

| Kategori | Parametre | Değer |
| :--- | :--- | :--- |
| **Retrieval** | `text_top_k` | `5` |
| **Retrieval** | `visual_top_k` | `2` |
| **Retrieval** | `threshold` | `0.35` |
| **Retrieval** | `deduplicate_parents` | `True` |
| **Generation** | `generator_model` | `openai/gpt-oss-120b (STRICT PROMPT)` |
| **RAGAS Judge** | `judge_model` | `gemini-3.1-flash-lite` |

## 🎯 Deterministik ID-Seviyesi Metrikler (LLM Bağımsız)

| Metrik Kümeleri | Metrik | Skor | Derece |
| :--- | :--- | :---: | :---: |
| **Child Chunk** | `det_child_recall` | **0.4350** | 🟠 ⭐⭐⭐☆☆ |
| **Child Chunk** | `det_child_precision` | **0.1900** | 🔴 ⭐⭐☆☆☆ |
| **Child Chunk** | `det_child_hit_rate` | **0.9500** | 🟢 ⭐⭐⭐⭐⭐ |
| **Child Chunk** | `det_child_mrr` | **0.8083** | 🟢 ⭐⭐⭐⭐⭐ |
| **Child Chunk** | `det_child_ndcg` | **0.4921** | 🟠 ⭐⭐⭐☆☆ |
| **Parent Chunk** | `det_parent_recall` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_precision` | **0.2000** | 🔴 ⭐⭐☆☆☆ |
| **Parent Chunk** | `det_parent_hit_rate` | **1.0000** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_mrr` | **0.8208** | 🟢 ⭐⭐⭐⭐⭐ |
| **Parent Chunk** | `det_parent_ndcg` | **0.8662** | 🟢 ⭐⭐⭐⭐⭐ |
| **Visual Chunk** | `det_visual_recall` | **0.0000** | ⚪ ☆☆☆☆☆ |

## ✨ Aggregate RAGAS Metrikleri

| Metrik Kategorisi | RAGAS Metrik | Skor | Derece |
| :--- | :--- | :---: | :---: |
| **Generation** | `faithfulness` | **0.9902** | 🟢 ⭐⭐⭐⭐⭐ |
| **Generation** | `answer_relevancy` | **0.8558** | 🟢 ⭐⭐⭐⭐⭐ |
| **Semantic Retrieval** | `llm_context_precision_with_reference` | **0.8083** | 🟢 ⭐⭐⭐⭐⭐ |
| **Semantic Retrieval** | `context_recall` | **0.9750** | 🟢 ⭐⭐⭐⭐⭐ |

## 🔍 Fault Type Kırılımı

| Fault Type | `answer_relevancy` | `context_recall` | `det_child_hit_rate` | `det_child_mrr` | `det_child_ndcg` | `det_child_precision` | `det_child_recall` | `det_parent_hit_rate` | `det_parent_mrr` | `det_parent_ndcg` | `det_parent_precision` | `det_parent_recall` | `det_visual_recall` | `faithfulness` | `llm_context_precision_with_reference` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **standards** | **0.8869** | **1.0000** | **1.0000** | **1.0000** | **0.4693** | **0.2000** | **0.3333** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **signal_processing** | **0.8317** | **1.0000** | **1.0000** | **0.5833** | **0.4007** | **0.2000** | **0.4583** | **1.0000** | **0.5833** | **0.6905** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **0.5833** |
| **unbalance** | **0.8615** | **1.0000** | **1.0000** | **0.5000** | **0.3869** | **0.2000** | **0.5000** | **1.0000** | **0.5000** | **0.6309** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **0.5000** |
| **misalignment** | **0.8051** | **1.0000** | **1.0000** | **1.0000** | **0.4693** | **0.2000** | **0.3333** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **looseness** | **0.8702** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **resonance** | **0.7349** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **0.7500** |
| **flow_hydrodynamic** | **0.8608** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **0.8667** | **1.0000** |
| **bearing_fault** | **0.8302** | **1.0000** | **1.0000** | **0.6667** | **0.3914** | **0.2000** | **0.3500** | **1.0000** | **0.6667** | **0.7500** | **0.2000** | **1.0000** | **0.0000** | **0.9688** | **0.6667** |
| **monitoring** | **0.9457** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **fundamentals** | **0.8397** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **eccentricity** | **0.8804** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **bent_shaft** | **0.9641** | **0.5000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **1.0000** | **0.2500** | **0.4307** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **0.2500** |
| **electrical** | **0.8543** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **aerodynamic** | **0.8539** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **journal_bearing** | **0.8810** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |
| **gear_fault** | **0.8901** | **1.0000** | **1.0000** | **1.0000** | **0.6131** | **0.2000** | **0.5000** | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |

## 📋 Soru Bazlı Detaylı Kıyaslama Tablosu

| ID | Soru | Kategori | Hit (Parent) | Beklenen Parent | Çekilen Parent | Context Durumu |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| `q_005` | ISO 10816 standartları ve titreşim şiddet tab... | `standards` | **✅ Başarılı** | `text_parent_15` | `text_parent_15, text_parent_17, text_parent_11, text_parent_13, text_parent_9` | 🟢 5 chunk |
| `q_009` | FFT analizinde pencereleme (Windowing / Leaka... | `signal_processing` | **✅ Başarılı** | `text_parent_21` | `text_parent_19, text_parent_21, text_parent_22, text_parent_42, text_parent_27` | 🟢 5 chunk |
| `q_014` | Rotorda dengesizlik (Unbalance) arızası spekt... | `unbalance` | **✅ Başarılı** | `text_parent_28` | `text_parent_34, text_parent_28, text_parent_33, text_parent_29, text_parent_43` | 🟢 5 chunk |
| `q_016` | Açısal eksen kaçıklığı (Angular Misalignment)... | `misalignment` | **✅ Başarılı** | `text_parent_31` | `text_parent_31, text_parent_30, text_parent_41, text_parent_49, text_parent_43` | 🟢 5 chunk |
| `q_017` | Mekanik gevşeklik (Mechanical Looseness) arız... | `looseness` | **✅ Başarılı** | `text_parent_32` | `text_parent_32, text_parent_38, text_parent_31, text_parent_43, text_parent_34` | 🟢 5 chunk |
| `q_018` | Rezonans (Resonance) ve doğal frekans tespiti... | `resonance` | **✅ Başarılı** | `text_parent_33` | `text_parent_33, text_parent_46, text_parent_32, text_parent_42, text_parent_34` | 🟢 5 chunk |
| `q_019` | Pompalarda kanat geçiş frekansı (Vane Pass Fr... | `flow_hydrodynamic` | **✅ Başarılı** | `text_parent_36` | `text_parent_36, text_parent_37, text_parent_33, text_parent_34, text_parent_35` | 🟢 5 chunk |
| `q_020` | Rulman arızalarının gelişme aşamalarında (Sta... | `bearing_fault` | **✅ Başarılı** | `text_parent_39` | `text_parent_27, text_parent_31, text_parent_39, text_parent_4, text_parent_40` | 🟢 5 chunk |
| `q_021` | Bir titreşim analiz sisteminin temel bileşenl... | `monitoring` | **✅ Başarılı** | `text_parent_7` | `text_parent_7, text_parent_42, text_parent_22, text_parent_5, text_parent_50` | 🟢 5 chunk |
| `q_022` | Titreşim hareketinde bir çevrim (one cycle of... | `fundamentals` | **✅ Başarılı** | `text_parent_12` | `text_parent_12, text_parent_47, text_parent_26, text_parent_13, text_parent_24` | 🟢 5 chunk |
| `q_023` | Analog ve dijital sinyal (Analog/Digital Sign... | `signal_processing` | **✅ Başarılı** | `text_parent_16` | `text_parent_16, text_parent_19, text_parent_17, text_parent_20, text_parent_7` | 🟢 5 chunk |
| `q_024` | FFT analizinde Örnek Sayısı (N), Kayıt Süresi... | `signal_processing` | **✅ Başarılı** | `text_parent_20` | `text_parent_19, text_parent_22, text_parent_20, text_parent_42, text_parent_47` | 🟢 5 chunk |
| `q_025` | Cepstrum analizi nedir ve hangi tür arızaları... | `signal_processing` | **✅ Başarılı** | `text_parent_26` | `text_parent_41, text_parent_26, text_parent_25, text_parent_27, text_parent_34` | 🟢 5 chunk |
| `q_026` | Eksantriklik (Eccentricity) arızası faz ölçüm... | `eccentricity` | **✅ Başarılı** | `text_parent_29` | `text_parent_29, text_parent_34, text_parent_28, text_parent_33, text_parent_43` | 🟢 5 chunk |
| `q_027` | Eğik şaft (Bent Shaft) arızasında eksenel tit... | `bent_shaft` | **✅ Başarılı** | `text_parent_30` | `text_parent_31, text_parent_29, text_parent_32, text_parent_30, text_parent_38` | 🟢 5 chunk |
| `q_028` | Titreşim spektrumunda elektriksel kaynaklı bi... | `electrical` | **✅ Başarılı** | `text_parent_34` | `text_parent_34, text_parent_27, text_parent_41, text_parent_22, text_parent_19` | 🟢 5 chunk |
| `q_029` | Fanlarda ve blowerlarda akış türbülansı (flow... | `aerodynamic` | **✅ Başarılı** | `text_parent_37` | `text_parent_37, text_parent_49, text_parent_36, text_parent_39, text_parent_38` | 🟢 5 chunk |
| `q_030` | Yağ girdabı (Oil Whirl) arızası hangi frekans... | `journal_bearing` | **✅ Başarılı** | `text_parent_38` | `text_parent_38, text_parent_39, text_parent_34, text_parent_18, text_parent_41` | 🟢 5 chunk |
| `q_031` | Rulman karakteristik arıza frekansları (BPFO,... | `bearing_fault` | **✅ Başarılı** | `text_parent_39` | `text_parent_39, text_parent_18, text_parent_34, text_parent_36, text_parent_28` | 🟢 5 chunk |
| `q_032` | Dişli kutusu (gearbox) titreşim spektrumunda ... | `gear_fault` | **✅ Başarılı** | `text_parent_41` | `text_parent_41, text_parent_49, text_parent_20, text_parent_23, text_parent_19` | 🟢 5 chunk |
