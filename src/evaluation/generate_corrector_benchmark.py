"""
VibraDiag — Corrector Benchmark Dataset Generator.

Generates a balanced evaluation dataset for testing the Self-Corrector node.
Modes:
- Default: dev8 (8 queries representing 8 specific scenarios)
- --full: 32 queries (every gold item matched with 4 scenarios)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from schemas.schemas import CorrectorBenchmarkItem


def load_json(filepath: Path) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_benchmark(full: bool = False) -> list[dict[str, Any]]:
    project_root = Path(__file__).resolve().parent.parent.parent
    dev_gold_path = project_root / "data" / "evaluation" / "retrieval_benchmark_dev8.json"
    text_chunks_path = project_root / "text_chunks.json"
    
    if full:
        output_path = project_root / "data" / "evaluation" / "corrector_benchmark.json"
    else:
        output_path = project_root / "data" / "evaluation" / "corrector_benchmark_dev8.json"

    gold_items = load_json(dev_gold_path)
    chunks_list = load_json(text_chunks_path)
    chunks_dict = {c["chunk_id"]: c.get("text", "") for c in chunks_list if "chunk_id" in c}

    benchmark_items: list[CorrectorBenchmarkItem] = []

    dsp_templates: dict[str, dict[str, Any]] = {
        "unbalance": {
            "primary_fault": "unbalance",
            "rpm": 1500.0,
            "running_speed_hz": 25.0,
            "dominant_frequency_hz": 25.0,
            "dominant_order": "1X",
            "phase_difference_deg": 90.0,
            "iso_severity_zone": "Zone C",
        },
        "misalignment": {
            "primary_fault": "angular_misalignment",
            "rpm": 1800.0,
            "running_speed_hz": 30.0,
            "dominant_frequency_hz": 60.0,
            "dominant_order": "2X",
            "axial_vibration_high": True,
            "phase_difference_deg": 180.0,
        },
        "bearing_fault": {
            "primary_fault": "inner_race_fault",
            "fault_type_abbr": "BPFI",
            "rpm": 1200.0,
            "bpfi_hz": 108.5,
            "bpfo_hz": 71.5,
            "bsf_hz": 48.2,
            "ftf_hz": 8.9,
            "dominant_frequency_hz": 108.5,
            "stage": "Stage 2",
        },
        "looseness": {
            "primary_fault": "mechanical_looseness",
            "rpm": 1500.0,
            "harmonics_present": ["1X", "2X", "3X", "4X", "0.5X"],
            "dominant_order": "Multiple Harmonics",
        },
        "standards": {
            "machine_class": "Class II",
            "vibration_velocity_rms_mm_s": 4.8,
            "iso_severity_zone": "Zone C",
            "iso_standard": "ISO 10816-1",
        },
        "signal_processing": {
            "windowing": "Hanning",
            "sampling_frequency_hz": 20480.0,
            "leakage_detected": False,
        },
        "flow_hydrodynamic": {
            "primary_fault": "vane_pass_vibration",
            "rpm": 1450.0,
            "n_vanes": 5,
            "vpf_hz": 120.8,
        },
        "resonance": {
            "natural_frequency_hz": 142.0,
            "test_type": "bump_test",
            "phase_shift_deg": 180.0,
        },
    }

    unrelated_chunk_ids = [
        "text_parent_1_child_0", 
        "text_parent_3_child_0",  
    ]

    if full:
        for item in gold_items:
            q_id = item["_id"]
            question = item["question"]
            gt_answer = item["ground_truth_answer"]
            fault_type = item.get("fault_type", "general")
            exp_chunk_ids = item.get("expected_chunk_ids", [])

            matched_chunks = [
                {"id": cid, "text": chunks_dict.get(cid, "Örnek teknik standart metni.")}
                for cid in exp_chunk_ids
            ]
            if not matched_chunks:
                matched_chunks = [{"id": f"dummy_{q_id}", "text": gt_answer}]

            dsp_info = dsp_templates.get(fault_type, {"fault_type": fault_type})

            clean_item = CorrectorBenchmarkItem(
                id=f"{q_id}_clean",
                scenario_type="clean",
                user_query=question,
                dsp_ground_truth=dsp_info,
                retrieved_contexts=matched_chunks,
                sub_queries=[question],
                candidate_response=gt_answer,
                expected_decision="approved",
                expected_failing_checkers=[],
                notes="Hatasız referans çıktı, onaylanmalı.",
            )
            benchmark_items.append(clean_item)

            if fault_type == "bearing_fault":
                mismatched_answer = (
                    f"Yapılan spektrum analizinde rulman dış bileziğinde arıza (BPFO) tespit edilmiştir "
                    f"ve baskın frekans 71.5 Hz olarak bulunmuştur. İç bilezik (BPFI) sağlamdır."
                )
                dsp_for_mismatch = dsp_templates["bearing_fault"]  
            elif fault_type == "unbalance":
                mismatched_answer = (
                    f"Titreşim analiz sonuçlarına göre makinede 2X frekansında paralel eksen kaçıklığı "
                    f"(Misalignment) mevcuttur. Dengesizlik (Unbalance) gözlenmemiştir."
                )
                dsp_for_mismatch = dsp_templates["unbalance"]
            elif fault_type == "misalignment":
                mismatched_answer = (
                    f"Sinyal incelemesinde baskın frekans 1X RPM olarak ölçülmüş olup arıza "
                    f"rotorda dengesizlik (Unbalance) olarak sınıflandırılmıştır."
                )
                dsp_for_mismatch = dsp_templates["misalignment"]
            elif fault_type == "standards":
                mismatched_answer = (
                    f"Titreşim RMS hız değeri 4.8 mm/s ölçülmüş olup makine ISO 10816 standartlarına "
                    f"göre en iyi durum olan Zone A (Yeni makine kabul sınırı) içerisindedir."
                )
                dsp_for_mismatch = dsp_templates["standards"]
            else:
                mismatched_answer = (
                    f"Spektrum analizinde baskın frekans 1X RPM olarak hesaplanmış olup "
                    f"tespit edilen arıza hidrodinamik kavitasyondur."
                )
                dsp_for_mismatch = dsp_info

            dsp_mismatch_item = CorrectorBenchmarkItem(
                id=f"{q_id}_dsp_mismatch",
                scenario_type="dsp_mismatch",
                user_query=question,
                dsp_ground_truth=dsp_for_mismatch,
                retrieved_contexts=matched_chunks,
                sub_queries=[question],
                candidate_response=mismatched_answer,
                expected_decision="retry_generation",
                expected_failing_checkers=["dsp_consistency"],
                notes="DSP ground truth ile çelişen arıza/değer iddiası.",
            )
            benchmark_items.append(dsp_mismatch_item)

            unrelated_chunks = [
                {"id": cid, "text": chunks_dict.get(cid, "Genel makine bakım prensipleri...")}
                for cid in unrelated_chunk_ids
            ]
            hallucinated_answer = (
                f"ISO 99999 ve NASA-VIB standardına göre bu makinede plazma kaynaklı rezonans dalgalanması "
                f"mevcuttur ve kuantum filtreleme yöntemiyle 0.01 mikron seviyesine çekilmelidir. "
                f"Ayrıca rotor sıcaklığının 450 dereceye ulaştığı belgelerle sabittir."
            )
            unfaithful_item = CorrectorBenchmarkItem(
                id=f"{q_id}_unfaithful",
                scenario_type="unfaithful",
                user_query=question,
                dsp_ground_truth=dsp_info,
                retrieved_contexts=unrelated_chunks,
                sub_queries=[question],
                candidate_response=hallucinated_answer,
                expected_decision="retry_retrieval",
                expected_failing_checkers=["faithfulness"],
                notes="Context ile desteklenmeyen uydurma standart ve halüsinasyon iddiaları.",
            )
            benchmark_items.append(unfaithful_item)

            sub_q1 = f"{question.split(' ve ')[0]} nedir?" if " ve " in question else f"{question} mekanizması nedir?"
            sub_q2 = "Bu arıza durumunda uygulanması gereken düzeltici aksiyon adımları nelerdir?"
            partial_answer = f"{gt_answer.split('.')[0]}. (Düzeltici aksiyonlar ve bakım adımlarına burada yer verilmemiştir)."

            sub_query_gap_item = CorrectorBenchmarkItem(
                id=f"{q_id}_sub_query_gap",
                scenario_type="sub_query_gap",
                user_query=f"{sub_q1} ve {sub_q2}",
                dsp_ground_truth=dsp_info,
                retrieved_contexts=matched_chunks,
                sub_queries=[sub_q1, sub_q2],
                candidate_response=partial_answer,
                expected_decision="retry_generation",
                expected_failing_checkers=["sub_query_relevance"],
                notes="Alt-sorulardan birini tamamen atlayan eksik yanıt.",
            )
            benchmark_items.append(sub_query_gap_item)
    else:
        for i, item in enumerate(gold_items):
            if i >= 8: break 
            
            q_id = item["_id"]
            question = item["question"]
            gt_answer = item["ground_truth_answer"]
            fault_type = item.get("fault_type", "general")
            exp_chunk_ids = item.get("expected_chunk_ids", [])

            matched_chunks = [
                {"id": cid, "text": chunks_dict.get(cid, "Örnek teknik standart metni.")}
                for cid in exp_chunk_ids
            ]
            if not matched_chunks:
                matched_chunks = [{"id": f"dummy_{q_id}", "text": gt_answer}]
            
            unrelated_chunks = [
                {"id": cid, "text": chunks_dict.get(cid, "Genel makine bakım prensipleri...")}
                for cid in unrelated_chunk_ids
            ]

            dsp_info = dsp_templates.get(fault_type, {"fault_type": fault_type})
            
            if i == 0:
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_clean",
                        scenario_type="clean",
                        user_query=question,
                        dsp_ground_truth=dsp_info,
                        retrieved_contexts=matched_chunks,
                        sub_queries=[question],
                        candidate_response=gt_answer,
                        expected_decision="approved",
                        expected_failing_checkers=[],
                        notes="Kolay Soru - Hatasız referans çıktı, onaylanmalı.",
                    )
                )
            elif i == 1:
                dsp_for_mismatch = dsp_templates.get("unbalance", {}) if fault_type != "unbalance" else dsp_templates.get("bearing_fault", {})
                mismatched_answer = (
                    "Titreşim analiz sonuçlarına göre makinede 2X frekansında paralel eksen kaçıklığı "
                    "(Misalignment) mevcuttur. Dengesizlik (Unbalance) gözlenmemiştir."
                )
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_dsp_mismatch",
                        scenario_type="dsp_mismatch",
                        user_query=question,
                        dsp_ground_truth=dsp_for_mismatch,
                        retrieved_contexts=matched_chunks,
                        sub_queries=[question],
                        candidate_response=mismatched_answer,
                        expected_decision="retry_generation",
                        expected_failing_checkers=["dsp_consistency"],
                        notes="Kolay Soru - DSP ground truth ile çelişen iddia.",
                    )
                )
            elif i == 2:
                sub_q1 = f"{question.split(' ve ')[0]} nedir?" if " ve " in question else f"{question} mekanizması nedir?"
                sub_q2 = "Bu arıza durumunda uygulanması gereken düzeltici aksiyon adımları nelerdir?"
                partial_answer = f"{gt_answer.split('.')[0]}. (Düzeltici aksiyonlar ve bakım adımlarına burada yer verilmemiştir)."
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_sub_query_gap",
                        scenario_type="sub_query_gap",
                        user_query=f"{sub_q1} ve {sub_q2}",
                        dsp_ground_truth=dsp_info,
                        retrieved_contexts=matched_chunks,
                        sub_queries=[sub_q1, sub_q2],
                        candidate_response=partial_answer,
                        expected_decision="retry_generation",
                        expected_failing_checkers=["sub_query_relevance"],
                        notes="Zor Soru - Alt-sorulardan birini tamamen atlayan eksik yanıt.",
                    )
                )
            elif i == 3:
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_clean_hard",
                        scenario_type="clean",
                        user_query=question,
                        dsp_ground_truth=dsp_info,
                        retrieved_contexts=matched_chunks,
                        sub_queries=[question],
                        candidate_response=gt_answer,
                        expected_decision="approved",
                        expected_failing_checkers=[],
                        notes="Zor Soru - Hatasız referans çıktı, onaylanmalı.",
                    )
                )
            elif i == 4:
                hallucinated_answer = (
                    f"Makinede plazma kaynaklı rezonans dalgalanması mevcuttur ve "
                    f"kuantum filtreleme yöntemiyle 0.01 mikron seviyesine çekilmelidir."
                )
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_unfaithful_1",
                        scenario_type="unfaithful",
                        user_query=question,
                        dsp_ground_truth=dsp_info,
                        retrieved_contexts=unrelated_chunks,
                        sub_queries=[question],
                        candidate_response=hallucinated_answer,
                        expected_decision="retry_retrieval",
                        expected_failing_checkers=["faithfulness"],
                        notes="Halüsinasyon - İlgisiz context ile üretilmiş uydurma çıktı.",
                    )
                )
            elif i == 5:
                hallucinated_answer = (
                    f"ISO 99999 ve NASA-VIB standardına göre bu makine kesinlikle kullanılmamalıdır. "
                    f"Rotor sıcaklığı 450 dereceyi geçmiştir."
                )
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_unfaithful_2",
                        scenario_type="unfaithful",
                        user_query=question,
                        dsp_ground_truth=dsp_info,
                        retrieved_contexts=unrelated_chunks,
                        sub_queries=[question],
                        candidate_response=hallucinated_answer,
                        expected_decision="retry_retrieval",
                        expected_failing_checkers=["faithfulness"],
                        notes="Halüsinasyon - İlgisiz context ile üretilmiş uydurma standart ve ölçüm iddiası.",
                    )
                )
            elif i == 6:
                refusal_answer = "Sağlanan metin parçalarında bu sorunun cevabını karşılayan bir bilgi bulunmamaktadır."
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_unanswerable_refusal",
                        scenario_type="clean",
                        user_query=question,
                        dsp_ground_truth=dsp_info,
                        retrieved_contexts=unrelated_chunks,
                        sub_queries=[question],
                        candidate_response=refusal_answer,
                        expected_decision="approved",
                        expected_failing_checkers=[],
                        notes="Cevapsız Durum - Doğru bir şekilde reddetme, onaylanmalı.",
                    )
                )
            elif i == 7:
                hallucinated_answer = "Eksik belgelere rağmen analizim gösteriyor ki sorun kavitasyon kaynaklıdır. RPM 1450 iken kavitasyon artar."
                benchmark_items.append(
                    CorrectorBenchmarkItem(
                        id=f"{q_id}_dev_unanswerable_hallucinated",
                        scenario_type="unfaithful",
                        user_query=question,
                        dsp_ground_truth=dsp_info,
                        retrieved_contexts=unrelated_chunks,
                        sub_queries=[question],
                        candidate_response=hallucinated_answer,
                        expected_decision="retry_retrieval",
                        expected_failing_checkers=["faithfulness"],
                        notes="Cevapsız Durum - Reddetmek yerine uydurma durumu.",
                    )
                )

    serializable = [item.model_dump() for item in benchmark_items]
    save_json(serializable, output_path)
    
    mode_str = "FULL 32-item test cases" if full else "DEV8 8-item dev cases"
    print(f"✅ Generated {len(benchmark_items)} corrector benchmark {mode_str} at: {output_path}")
    print(f"   - Clean cases: {sum(1 for x in benchmark_items if x.scenario_type == 'clean')}")
    print(f"   - DSP Mismatch: {sum(1 for x in benchmark_items if x.scenario_type == 'dsp_mismatch')}")
    print(f"   - Unfaithful: {sum(1 for x in benchmark_items if x.scenario_type == 'unfaithful')}")
    print(f"   - Sub-query Gap: {sum(1 for x in benchmark_items if x.scenario_type == 'sub_query_gap')}")
    return serializable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Corrector Benchmark")
    parser.add_argument("--full", action="store_true", help="Generate the full 32-item benchmark instead of dev8")
    args = parser.parse_args()
    
    generate_benchmark(full=args.full)
