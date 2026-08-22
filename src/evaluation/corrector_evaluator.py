"""
VibraDiag — Self-Corrector Offline Evaluator & Threshold Calibrator.

Evaluates the Self-Corrector against `corrector_benchmark.json`, computing:
1. Decision Confusion Matrix & Overall Accuracy
2. False Positive Rate on Clean Cases (Gereksiz Retry / Maliyet Riski)
3. Per-Checker Confusion Matrix, Precision, Recall & F1 (DSP, Faithfulness, Relevance)
4. Fast Score-Cached Threshold Calibration (Grid Search on F1-Score & Clean FPR)
5. Multi-Cycle Retry Session Simulation (Strategy escalation & Max cycles cap)
6. Cumulative Token Usage and Total Cost in USD
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

from schemas.schemas import CorrectorBenchmarkItem
from self_corrector.checkers import (
    DSPConsistencyChecker,
    FaithfulnessChecker,
    GeminiJudgeClient,
    RelevanceChecker,
)
from self_corrector.corrector_node import self_corrector_node

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("corrector_eval")


def run_evaluation(
    benchmark_path: Path,
    faithfulness_th: float = 0.50,
    relevance_th: float = 0.50,
    dsp_th: float = 1.0,
    sample_size: int | None = None,
    judge_client: Any | None = None,
    groq_client: Any | None = None,
) -> dict[str, Any]:
    """
    Evaluates the Self-Corrector on a benchmark dataset using specified thresholds.
    Wires thresholds directly to self_corrector_node via thresholds_override.
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if sample_size:
        raw_items = raw_items[:sample_size]

    total_samples = len(raw_items)
    logger.info("Running evaluation on %d benchmark samples with thresholds (faith=%.2f, rel=%.2f, dsp=%.2f)...",
                total_samples, faithfulness_th, relevance_th, dsp_th)

    thresholds_override = {
        "faithfulness_threshold": faithfulness_th,
        "relevance_threshold": relevance_th,
        "dsp_consistency_threshold": dsp_th,
    }

    results_by_scenario: dict[str, dict[str, int]] = {
        "clean": {"total": 0, "correct": 0, "false_positives": 0},
        "dsp_mismatch": {"total": 0, "correct": 0, "missed": 0},
        "unfaithful": {"total": 0, "correct": 0, "missed": 0},
        "sub_query_gap": {"total": 0, "correct": 0, "missed": 0},
    }

    # Per-checker confusion metrics: TP, FP, FN, TN
    # TP: Checker failed when expected to fail
    # FP: Checker failed when NOT expected to fail
    # FN: Checker passed when expected to fail
    # TN: Checker passed when NOT expected to fail
    #
    # NOTE on Single vs Multi-Label Cascading Evaluation:
    # DSP fast-fail olduğunda faithfulness/sub_query_relevance hiç çalışmıyor, bu durumda failed=False varsayılıyor.
    # Bu varsayım tek-etiketli (single-fault) senaryolar için matematiksel olarak doğrudur.
    # İleride hem DSP hem Faithfulness'ın aynı anda fail etmesi beklenen çok-etiketli (multi-label) senaryolar
    # eklenirse, fast-fail nedeniyle atlanan checker'ların TN/FN değerlendirmesi gözden geçirilmelidir.
    checker_cm: dict[str, dict[str, int]] = {
        "dsp_consistency": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "faithfulness": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "sub_query_relevance": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
    }

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost_usd = 0.0
    total_latency_ms = 0.0

    eval_details = []

    for item_raw in raw_items:
        item = CorrectorBenchmarkItem(**item_raw)
        stype = item.scenario_type
        results_by_scenario[stype]["total"] += 1

        state = {
            "session_id": f"eval_{item.id}",
            "user_query": item.user_query,
            "llm_response": item.candidate_response,
            "signal_analysis": item.dsp_ground_truth,
            "text_passages": item.retrieved_contexts,
            "sub_queries": item.sub_queries,
            "correction_attempts": {"retrieval": 0, "generation": 0, "total": 0},
            "excluded_chunk_ids": [],
            "max_text_score": 0.85 if stype != "unfaithful" else 0.20,
        }

        start_t = time.time()
        node_output = self_corrector_node(
            state=state,
            thresholds_override=thresholds_override,
            judge_client=judge_client,
            groq_client=groq_client,
        )
        latency = (time.time() - start_t) * 1000
        total_latency_ms += latency

        route = node_output.get("route_decision", "approved")
        is_correct = False

        if stype == "clean":
            if route == "approved":
                is_correct = True
                results_by_scenario[stype]["correct"] += 1
            else:
                results_by_scenario[stype]["false_positives"] += 1

        elif stype == "dsp_mismatch":
            if route == "retry_generation":
                is_correct = True
                results_by_scenario[stype]["correct"] += 1
            else:
                results_by_scenario[stype]["missed"] += 1

        elif stype == "unfaithful":
            if route == "retry_retrieval":
                is_correct = True
                results_by_scenario[stype]["correct"] += 1
            else:
                results_by_scenario[stype]["missed"] += 1

        elif stype == "sub_query_gap":
            if route in ("retry_retrieval", "retry_generation"):
                is_correct = True
                results_by_scenario[stype]["correct"] += 1
            else:
                results_by_scenario[stype]["missed"] += 1

        item_checker_results = node_output.get("checker_results", {})
        expected_failing = set(item.expected_failing_checkers)

        for c_name in ("dsp_consistency", "faithfulness", "sub_query_relevance"):
            c_res = item_checker_results.get(c_name)

            if c_res is not None:
                passed = c_res.get("passed", True) if isinstance(c_res, dict) else getattr(c_res, "passed", True)
                failed = not passed
            else:
                failed = False

            exp_fail = c_name in expected_failing

            if exp_fail and failed:
                checker_cm[c_name]["TP"] += 1
            elif not exp_fail and failed:
                checker_cm[c_name]["FP"] += 1
            elif exp_fail and not failed:
                checker_cm[c_name]["FN"] += 1
            else:
                checker_cm[c_name]["TN"] += 1

        metrics = node_output.get("corrector_metrics") or {}
        total_tokens_in += metrics.get("tokens_input", 0)
        total_tokens_out += metrics.get("tokens_output", 0)
        total_cost_usd += metrics.get("estimated_cost_usd", 0.0)

        eval_details.append({
            "id": item.id,
            "scenario": stype,
            "expected": item.expected_decision,
            "expected_failing_checkers": item.expected_failing_checkers,
            "actual": route,
            "is_flagged": node_output.get("is_flagged", False),
            "flag_reason": node_output.get("flag_reason"),
            "checker_results": item_checker_results,
            "is_correct": is_correct,
            "latency_ms": round(latency, 2),
        })

    total_correct = sum(v["correct"] for v in results_by_scenario.values())
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0

    clean_total = results_by_scenario["clean"]["total"]
    fp_rate = (results_by_scenario["clean"]["false_positives"] / clean_total) if clean_total > 0 else 0.0

    fail_open_count = sum(
        1 for d in eval_details if d.get("is_flagged") and d.get("actual") == "approved"
    )

    checker_metrics: dict[str, dict[str, Any]] = {}
    for c_name, counts in checker_cm.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        tn = counts["TN"]

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        checker_metrics[c_name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    report = {
        "summary": {
            "total_samples": total_samples,
            "overall_accuracy": round(accuracy, 4),
            "clean_false_positive_rate": round(fp_rate, 4),
            "fail_open_approved_count": fail_open_count,
            "thresholds_used": thresholds_override,
            "scenario_breakdown": results_by_scenario,
            "checker_metrics": checker_metrics,
            "tokens": {
                "input": total_tokens_in,
                "output": total_tokens_out,
                "total": total_tokens_in + total_tokens_out,
            },
            "estimated_cost_usd": round(total_cost_usd, 6),
            "avg_latency_ms": round(total_latency_ms / total_samples, 2) if total_samples > 0 else 0.0,
        },
        "details": eval_details,
    }
    return report


def calibrate_thresholds(
    benchmark_path: Path,
    faithfulness_range: list[float] | None = None,
    relevance_range: list[float] | None = None,
    dsp_range: list[float] | None = None,
    fp_penalty_weight: float = 1.5,
    sample_size: int | None = None,
    judge_client: Any | None = None,
    groq_client: Any | None = None,
) -> dict[str, Any]:
    """
    Executes fast score-cached Grid Search calibration across threshold combinations.
    Evaluates raw checker scores once, then performs in-memory sweep in milliseconds (0 tokens).

    Note:
    DSP Consistency Checker deterministic bir pass/fail üretir, threshold kalibrasyonu kapsamı dışındadır (dsp_grid sabit [1.0]).
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if sample_size:
        raw_items = raw_items[:sample_size]

    faith_grid = faithfulness_range or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    rel_grid = relevance_range or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    dsp_grid = dsp_range or [1.0]

    logger.info("Starting Grid Search calibration: Collecting raw scores for %d benchmark samples...", len(raw_items))

    cached_scores: list[dict[str, Any]] = []

    faith_checker = FaithfulnessChecker(judge_client=judge_client)
    rel_checker = RelevanceChecker(judge_client=judge_client)

    for item_raw in raw_items:
        item = CorrectorBenchmarkItem(**item_raw)

        dsp_res = DSPConsistencyChecker.check(
            generated_text=item.candidate_response,
            signal_data=item.dsp_ground_truth,
        )

        faith_res = faith_checker.check(
            generated_text=item.candidate_response,
            context_chunks=item.retrieved_contexts,
            threshold=0.0,  
        )

        rel_res = rel_checker.check(
            user_query=item.user_query,
            generated_text=item.candidate_response,
            sub_queries=item.sub_queries,
            threshold=0.0, 
        )

        cached_scores.append({
            "id": item.id,
            "scenario": item.scenario_type,
            "expected_decision": item.expected_decision,
            "expected_failing": set(item.expected_failing_checkers),
            "dsp_passed": dsp_res.passed,
            "dsp_score": dsp_res.score,
            "faith_score": faith_res.score,
            "rel_score": rel_res.score,
        })

    logger.info("Raw scores collected. Sweeping %d threshold combinations in memory...",
                len(faith_grid) * len(rel_grid) * len(dsp_grid))

    calibration_results = []
    total_clean = sum(1 for x in cached_scores if x["scenario"] == "clean")

    for f_th, r_th, d_th in itertools.product(faith_grid, rel_grid, dsp_grid):
        correct_count = 0
        clean_fp = 0

        tp_all = 0
        fp_all = 0
        fn_all = 0
        tn_all = 0

        for sample in cached_scores:
            dsp_ok = sample["dsp_passed"]
            faith_ok = sample["faith_score"] >= f_th
            rel_ok = sample["rel_score"] >= r_th

            if not dsp_ok:
                sim_route = "retry_generation"
            elif not faith_ok or not rel_ok:
                sim_route = "retry_retrieval"
            else:
                sim_route = "approved"

            stype = sample["scenario"]
            is_correct = False

            if stype == "clean":
                if sim_route == "approved":
                    is_correct = True
                else:
                    clean_fp += 1
            elif stype == "dsp_mismatch":
                if sim_route == "retry_generation":
                    is_correct = True
            elif stype == "unfaithful":
                if sim_route == "retry_retrieval":
                    is_correct = True
            elif stype == "sub_query_gap":
                if sim_route in ("retry_retrieval", "retry_generation"):
                    is_correct = True

            if is_correct:
                correct_count += 1

            actual_needs_correction = (sim_route != "approved")
            expected_needs_correction = (sample["expected_decision"] != "approved")

            if expected_needs_correction and actual_needs_correction:
                tp_all += 1
            elif not expected_needs_correction and actual_needs_correction:
                fp_all += 1
            elif expected_needs_correction and not actual_needs_correction:
                fn_all += 1
            else:
                tn_all += 1

        accuracy = correct_count / len(cached_scores)
        clean_fpr = (clean_fp / total_clean) if total_clean > 0 else 0.0

        prec = (tp_all / (tp_all + fp_all)) if (tp_all + fp_all) > 0 else 0.0
        rec = (tp_all / (tp_all + fn_all)) if (tp_all + fn_all) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        beta = 0.5
        f05 = ((1 + beta**2) * prec * rec / ((beta**2 * prec) + rec)) if ((beta**2 * prec) + rec) > 0 else 0.0


        utility_score = f1 - (fp_penalty_weight * clean_fpr)

        calibration_results.append({
            "thresholds": {
                "faithfulness_threshold": f_th,
                "relevance_threshold": r_th,
                "dsp_consistency_threshold": d_th,
            },
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "f05_score": round(f05, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "clean_false_positive_rate": round(clean_fpr, 4),
            "utility_score": round(utility_score, 4),
        })

    calibration_results.sort(key=lambda x: (x["utility_score"], x["f1_score"], x["accuracy"]), reverse=True)
    best_config = calibration_results[0]

    return {
        "fp_penalty_weight_used": fp_penalty_weight,
        "dsp_threshold_note": "DSP Consistency Checker deterministic bir pass/fail üretir, threshold kalibrasyonu kapsamı dışındadır (dsp_grid sabit [1.0]).",
        "best_config": best_config,
        "top_5_configurations": calibration_results[:5],
        "total_combinations_tested": len(calibration_results),
    }


def simulate_multi_cycle_session(
    benchmark_path: Path,
    sample_size: int | None = None,
    global_max_cycles: int = 3,
    judge_client: Any | None = None,
    groq_client: Any | None = None,
) -> dict[str, Any]:
    """
    Simulates multi-cycle session retry loops for benchmark samples until resolution or global_max_cycles.
    Monitors strategy escalation (domain_dictionary -> PRF -> LLM rewrite) and cycle convergence.
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if sample_size:
        raw_items = raw_items[:sample_size]

    logger.info("Running Multi-Cycle Session Simulation on %d samples (max_cycles=%d)...", len(raw_items), global_max_cycles)

    strategy_counts = {
        "domain_dictionary": 0,
        "pseudo_relevance_feedback": 0,
        "llm_query_rewriter": 0,
    }

    session_summaries = []
    total_cycles_all = 0
    resolved_count = 0
    flagged_max_cycles_count = 0

    for item_raw in raw_items:
        item = CorrectorBenchmarkItem(**item_raw)
        stype = item.scenario_type


        state: dict[str, Any] = {
            "session_id": f"sim_{item.id}",
            "user_query": item.user_query,
            "llm_response": item.candidate_response,
            "signal_analysis": item.dsp_ground_truth,
            "text_passages": item.retrieved_contexts,
            "sub_queries": item.sub_queries,
            "correction_attempts": {"retrieval": 0, "generation": 0, "total": 0},
            "excluded_chunk_ids": [],
            "max_text_score": 0.85 if stype != "unfaithful" else 0.20,
        }

        cycle_history = []
        final_route = "approved"

        for cycle in range(1, global_max_cycles + 2):
            output = self_corrector_node(
                state=state,
                judge_client=judge_client,
                groq_client=groq_client,
            )
            route = output.get("route_decision", "approved")
            final_route = route


            for h in output.get("corrector_history", []):
                strat = h.get("strategy")
                if strat in strategy_counts:
                    strategy_counts[strat] += 1
                cycle_history.append(h)

            if route in ("approved", "approved_with_warning"):
                break

            state["correction_attempts"] = output.get("correction_attempts", state["correction_attempts"])
            if output.get("excluded_chunk_ids"):
                state["excluded_chunk_ids"] = list(set(state["excluded_chunk_ids"] + output["excluded_chunk_ids"]))
            if output.get("reformulated_query"):
                state["user_query"] = output["reformulated_query"]

            # Mock Re-generation / Retrieval Resolution for Offline Simulation:
            if route == "retry_generation":
                if stype == "dsp_mismatch":
                    primary = (state.get("signal_analysis") or {}).get("primary_fault", "unbalance")
                    state["llm_response"] = (
                        f"DSP sinyal analizine ve spektrum bulgularına göre baskın arıza "
                        f"rotorda {primary} olarak doğrulanmıştır."
                    )
                elif stype == "sub_query_gap":
                    state["llm_response"] = (
                        f"{item.candidate_response.replace('(Düzeltici aksiyonlar ve bakım adımlarına burada yer verilmemiştir).', '')} "
                        f"Düzeltici aksiyon olarak: Balans ayarı yapılmalı, ağırlık optimizasyonu ve hizalama kontrolleri gerçekleştirilmelidir."
                    )
            elif route == "retry_retrieval":
                if stype == "unfaithful":
                    if "unanswerable" in item.id:
                        state["llm_response"] = "Sağlanan metin parçalarında bu sorunun cevabını karşılayan bir bilgi bulunmamaktadır."
                    else:
                        state["text_passages"] = [{"id": f"ctx_{item.id}", "text": f"{item.user_query} ile ilgili teknik standartlar ve mekanizma açıklamaları."}]
                        state["llm_response"] = f"{item.user_query} konusu bağlamdaki teknik standartlara tam olarak uygundur."

        total_cycles_used = len(cycle_history) or 1
        total_cycles_all += total_cycles_used

        if final_route == "approved":
            resolved_count += 1
        elif final_route == "approved_with_warning":
            flagged_max_cycles_count += 1
            current_total = state.get("correction_attempts", {}).get("total", 0)
            assert current_total >= global_max_cycles, (
                f"Cap enforcement did not trigger naturally for {item.id}. "
                f"Expected total attempts >= {global_max_cycles}, got {current_total}."
            )

        session_summaries.append({
            "id": item.id,
            "scenario": stype,
            "cycles_used": total_cycles_used,
            "final_route": final_route,
            "cycle_history": cycle_history,
        })

    avg_cycles = (total_cycles_all / len(raw_items)) if raw_items else 0.0

    return {
        "total_sessions": len(raw_items),
        "resolved_clean_or_fixed": resolved_count,
        "flagged_max_cycles_count": flagged_max_cycles_count,
        "avg_cycles_to_resolution": round(avg_cycles, 2),
        "strategy_trigger_distribution": strategy_counts,
        "sessions": session_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VibraDiag Self-Corrector on Benchmark Dataset")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/evaluation/corrector_benchmark.json",
        help="Path to benchmark JSON",
    )
    parser.add_argument("--sample", type=int, default=None, help="Sample count limit")
    parser.add_argument("--faithfulness-th", type=float, default=0.50)
    parser.add_argument("--relevance-th", type=float, default=0.50)
    parser.add_argument("--dsp-th", type=float, default=1.0)
    parser.add_argument(
        "--fp-penalty-weight",
        type=float,
        default=1.5,
        help="Penalty weight applied to Clean False Positive Rate in utility score calculation (default: 1.5)",
    )
    parser.add_argument("--calibrate", action="store_true", help="Run Grid Search threshold calibration")
    parser.add_argument("--simulate-cycles", action="store_true", help="Run multi-cycle retry session simulation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    bench_path = project_root / args.benchmark

    if not bench_path.exists():
        logger.info("Benchmark file not found: %s. Generating now...", bench_path)
        from evaluation.generate_corrector_benchmark import generate_benchmark
        generate_benchmark()

    out_dir = project_root / "data" / "eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Single Evaluation Run
    report = run_evaluation(
        benchmark_path=bench_path,
        faithfulness_th=args.faithfulness_th,
        relevance_th=args.relevance_th,
        dsp_th=args.dsp_th,
        sample_size=args.sample,
    )

    report_file = out_dir / "corrector_eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = report["summary"]
    print("\n" + "=" * 65)
    print("🎯 VIBRADIAG SELF-CORRECTOR EVALUATION REPORT")
    print("=" * 65)
    print(f"Toplam Test Örneği         : {summary['total_samples']}")
    print(f"Genel Doğruluk (Accuracy)   : {summary['overall_accuracy']:.2%}")
    print(f"False Positive Oranı (FP)  : {summary['clean_false_positive_rate']:.2%} (Gereksiz Retry / Maliyet Riski)")
    if summary.get("fail_open_approved_count", 0) > 0:
        print(f"Fail-Open Onay Sayısı       : {summary['fail_open_approved_count']} (LLM Hatası Toleransı)")
    print(f"Ortalama Gecikme (Latency) : {summary['avg_latency_ms']} ms")
    print(f"Toplam Token Tüketimi      : {summary['tokens']['total']} (In: {summary['tokens']['input']}, Out: {summary['tokens']['output']})")
    print(f"Tahmini Toplam Maliyet     : ${summary['estimated_cost_usd']:.6f} USD")
    print("\n--- Senaryo Bazlı Başarı ---")
    for sc, stats in summary["scenario_breakdown"].items():
        corr = stats["correct"]
        tot = stats["total"]
        pct = (corr / tot * 100) if tot > 0 else 0
        print(f"  • {sc:15s}: {corr}/{tot} (%{pct:.1f})")

    print("\n--- Checker Bazlı Performans (Confusion Matrix & Recall) ---")
    for c_name, m in summary["checker_metrics"].items():
        print(f"  • {c_name:22s} | Recall: {m['recall']:.2%} | Precision: {m['precision']:.2%} | F1: {m['f1']:.2%} (TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, TN={m['TN']})")

    if args.calibrate:
        print("\n" + "=" * 65)
        print("🔍 RUNNING THRESHOLD GRID SEARCH CALIBRATION")
        print("=" * 65)
        calib = calibrate_thresholds(
            benchmark_path=bench_path,
            sample_size=args.sample,
            fp_penalty_weight=args.fp_penalty_weight,
        )
        best = calib["best_config"]

        calib_file = out_dir / "corrector_calibration_report.json"
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump(calib, f, ensure_ascii=False, indent=2)

        print(f"Test Edilen Kombinasyon : {calib['total_combinations_tested']}")
        print(f"En İyi Eşik Konfigürasyonu:")
        print(f"  • Faithfulness Eşiği : {best['thresholds']['faithfulness_threshold']}")
        print(f"  • Relevance Eşiği     : {best['thresholds']['relevance_threshold']}")
        print(f"  • DSP Eşiği           : {best['thresholds']['dsp_consistency_threshold']} (Deterministik / Kalibrasyon dışı)")
        print(f"  • Beklenen F1-Skoru   : {best['f1_score']:.2%}")
        print(f"  • Beklenen Clean FPR : {best['clean_false_positive_rate']:.2%}")
        print(f"  • Utility Skoru       : {best['utility_score']:.4f} (FP Penalty: {calib.get('fp_penalty_weight_used', 1.5)})")
        print(f"  • Genel Doğruluk      : {best['accuracy']:.2%}")
        print(f"Kalibrasyon raporu kaydedildi: {calib_file}")

    if args.simulate_cycles:
        print("\n" + "=" * 65)
        print("🔄 RUNNING MULTI-CYCLE RETRY SIMULATION")
        print("=" * 65)
        sim = simulate_multi_cycle_session(benchmark_path=bench_path, sample_size=args.sample)
        sim_file = out_dir / "corrector_simulation_report.json"
        with open(sim_file, "w", encoding="utf-8") as f:
            json.dump(sim, f, ensure_ascii=False, indent=2)

        print(f"Toplam Simüle Oturum       : {sim['total_sessions']}")
        print(f"Ortalama Çözüm Döngüsü     : {sim['avg_cycles_to_resolution']} cycles")
        print(f"Maksimum Döngü Aşan (Flag) : {sim['flagged_max_cycles_count']}")
        print("Strateji Tetiklenme Dağılımı:")
        for s_name, count in sim["strategy_trigger_distribution"].items():
            print(f"  • {s_name:28s}: {count}")
        print(f"Simülasyon raporu kaydedildi: {sim_file}")

    print("=" * 65)
    print(f"Detaylı rapor kaydedildi: {report_file}\n")


if __name__ == "__main__":
    main()
