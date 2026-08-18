"""
VibraDiag — Central Orchestration Graph (Main Graph).
=====================================================

Combines:
1. Polymorphic Signal Ingestion (.wav, .mat, .csv).
2. Deterministic Signal Processing SubGraph (Kurtogram, Rotating/Reciprocating Expert, ISO 2372 / VDI 2056).
3. Diagnostic Plot Generation (FFT Spectrum, Kurtogram Heatmap).
4. Hybrid Vector & Decomposed RAG Retrieval (Dense + BM42 Sparse + Parent-Child DocStore + Reranker).
5. Groq Llama-3.3-70b Generation Engine (with ISO Zone D Emergency Directive & Signal Injection).
6. Multi-tiered Self-Corrector Verification Loop (DSP Fast-Fail, Gemini Faithfulness/Relevance, Dynamic Query Reformulation).
7. LangGraph State Persistence (MemorySaver Checkpointer support).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from deterministic_tools.reader import LoadedSignal, SignalReaderFactory
from deterministic_tools.signal_processing_subgraph import (
    plot_diagnostics_node,
    signal_processing_subgraph_node,
)
from generation.generator import generation_node
from generation.prompt_builder import build_initial_signal_query
from retrieval.retriever import RetrievalGraph
from schemas.states import VibraDiagMainState
from self_corrector.corrector_node import self_corrector_node

logger = logging.getLogger(__name__)

_default_retrieval_graph: RetrievalGraph | None = None


def get_default_retrieval_graph() -> RetrievalGraph:
    """Lazy initializer for default RetrievalGraph instance."""
    global _default_retrieval_graph
    if _default_retrieval_graph is None:
        from config_loader import load_appcfg, load_retcfg
        from retrieval import (
            DocStore,
            Embedder,
            Reranker,
            TextChildQdrantDB,
            TextRetriever,
            VisualQdrantDB,
            VisualRetriever,
        )
        from retrieval.sparse_encoder import SparseEncoder
        from query_decomposer.classifier import QueryComplexityClassifier
        from query_decomposer.llm_decomposer import LLMDecomposer
        from qdrant_client import QdrantClient

        try:
            app_cfg = load_appcfg()
            ret_cfg = load_retcfg()
            embed_model = getattr(app_cfg, "embeddings", {}).get("model", "BAAI/bge-m3")
            device = getattr(app_cfg, "embeddings", {}).get("device", "mps")
            reranker_cfg = getattr(ret_cfg, "hybrid_search", {}).get("reranker", {})
            rerank_model = reranker_cfg.get("model", "BAAI/bge-reranker-large")
            threshold = reranker_cfg.get("score_threshold", 0.35)
            qdrant_path = getattr(app_cfg, "paths", {}).get("qdrant_persist_path", "./qdrant_data")
            docstore_path = getattr(app_cfg, "paths", {}).get("docstore_path", "./docstore.db")
        except Exception as e:
            logger.warning("Failed to load config for RetrievalGraph, using defaults: %s", e)
            embed_model = "BAAI/bge-m3"
            device = "cpu"
            rerank_model = "BAAI/bge-reranker-large"
            threshold = 0.35
            qdrant_path = "./qdrant_data"
            docstore_path = "./docstore.db"

        embedder = Embedder(model_name=embed_model, device=device)
        sparse_encoder = SparseEncoder()
        docstore = DocStore(persist_path=docstore_path)
        qdrant_client = QdrantClient(path=qdrant_path)
        text_vector_db = TextChildQdrantDB(client=qdrant_client, sparse_encoder=sparse_encoder)
        visual_vector_db = VisualQdrantDB(client=qdrant_client)
        reranker = Reranker(model_name=rerank_model, device=device, score_threshold=threshold)

        text_retriever = TextRetriever(
            embedder=embedder,
            vector_db=text_vector_db,
            docstore=docstore,
            reranker=reranker,
            sparse_encoder=sparse_encoder,
        )
        visual_retriever = VisualRetriever(
            embedder=embedder,
            vector_db=visual_vector_db,
        )

        classifier = None
        decomposer = None
        try:
            classifier = QueryComplexityClassifier()
            decomposer = LLMDecomposer()
        except Exception as e:
            logger.warning("Could not initialize classifier/decomposer for RetrievalGraph: %s", e)

        _default_retrieval_graph = RetrievalGraph(
            text_retriever=text_retriever,
            visual_retriever=visual_retriever,
            threshold=threshold,
            classifier=classifier,
            decomposer=decomposer,
        )

    return _default_retrieval_graph

def input_router_node(state: VibraDiagMainState) -> dict[str, Any]:
    """
    Analyzes input modalities (signal file path, user text query, message history)
    and initializes route decisions and execution flags.
    """
    user_query = (state.get("user_query") or "").strip()
    signal_file = state.get("signal_file_path")

    messages = state.get("messages") or []
    if not user_query and messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                user_query = str(msg.content).strip()
                break

    has_signal = bool(signal_file and str(signal_file).strip())
    has_query = bool(user_query)

    if has_signal and has_query:
        route_decision = "hybrid"
        needs_signal_context = True
    elif has_signal and not has_query:
        route_decision = "signal_only"
        needs_signal_context = True
    elif not has_signal and has_query:
        route_decision = "rag_only"
        needs_signal_context = False
    else:
        route_decision = "error"
        needs_signal_context = False

    logger.info(
        "InputRouter: route_decision='%s', has_signal=%s, has_query=%s",
        route_decision,
        has_signal,
        has_query,
    )

    update: dict[str, Any] = {
        "user_query": user_query,
        "route_decision": route_decision,
        "needs_signal_context": needs_signal_context,
    }

    if route_decision == "error":
        err_msg = "Geçerli bir kullanıcı sorusu veya analiz edilecek bir sinyal dosyası gereklidir."
        update["errors"] = [err_msg]
        update["llm_response"] = f"Hata: {err_msg}"

    return update


def signal_loader_node(state: VibraDiagMainState) -> dict[str, Any]:
    """
    Loads raw vibration signal from disk using SignalReaderFactory (.wav, .mat, .csv).
    """
    signal_file = state.get("signal_file_path")
    if not signal_file:
        return {"errors": ["Sinyal dosyası yolu belirtilmedi."], "route_decision": "error"}

    meta = state.get("machine_metadata") or {}
    expected_fs = meta.get("fs") or meta.get("expected_fs")

    try:
        loaded: LoadedSignal = SignalReaderFactory.load(str(signal_file), expected_fs=expected_fs)
        logger.info(
            "SignalLoader: Successfully loaded '%s' (channels=%s, fs=%.1f Hz, rpm=%s)",
            signal_file,
            list(loaded.channels.keys()),
            loaded.fs,
            loaded.rpm,
        )

        updated_meta = dict(meta)
        if loaded.rpm and "rpm" not in updated_meta:
            updated_meta["rpm"] = float(loaded.rpm)

        # Sanitize metadata dictionary to standard Python primitives (no numpy scalar types)
        sanitized_meta = {}
        for k, v in updated_meta.items():
            if hasattr(v, "item"):
                sanitized_meta[k] = v.item()
            elif isinstance(v, (float, int, str, bool, list, dict)) or v is None:
                sanitized_meta[k] = v
            else:
                sanitized_meta[k] = str(v)

        result: dict[str, Any] = {
            "machine_metadata": sanitized_meta,
        }

        return result

    except Exception as e:
        logger.error("SignalLoader failed for '%s': %s", signal_file, e, exc_info=True)
        return {
            "errors": [f"Sinyal dosyası yüklenemedi ({signal_file}): {e}"],
            "route_decision": "error",
            "llm_response": f"Sinyal dosyası okuma hatası: {e}",
        }


def signal_dsp_node(state: VibraDiagMainState) -> dict[str, Any]:
    """
    Executes the isolated SignalProcessingSubGraph.
    """
    errors = state.get("errors") or []
    if errors:
        logger.warning("Skipping signal_dsp_node due to existing errors: %s", errors)
        return {}

    logger.info("Executing SignalProcessingSubGraph node...")
    
    # Ensure loaded_signal is populated
    active_state = dict(state)
    if "loaded_signal" not in active_state or active_state["loaded_signal"] is None:
        signal_file = active_state.get("signal_file_path")
        if signal_file:
            meta = active_state.get("machine_metadata") or {}
            expected_fs = meta.get("fs") or meta.get("expected_fs")
            active_state["loaded_signal"] = SignalReaderFactory.load(str(signal_file), expected_fs=expected_fs)

    dsp_output = signal_processing_subgraph_node(active_state)
    sig_res = dsp_output.get("signal_processing_result", {})
    if sig_res:
        from deterministic_tools.fault_analyzer import pick_primary_fault
        fault_summary = pick_primary_fault(
            direct_spectrum_diagnosis=sig_res.get("direct_spectrum_diagnosis"),
            envelope_diagnosis=sig_res.get("envelope_diagnosis"),
            reciprocating_diagnosis=sig_res.get("reciprocating_diagnosis"),
        )
        dsp_output["primary_fault"] = fault_summary.get("primary_fault_name")
        dsp_output["primary_fault_data"] = fault_summary
        if "time_domain_stats" in sig_res:
            dsp_output["time_domain_stats"] = sig_res["time_domain_stats"]

    return dsp_output


def diagnostic_plots_wrapper_node(state: VibraDiagMainState) -> dict[str, Any]:
    """
    Generates FFT Spectrum and Kurtogram Heatmap Plotly figures.
    """
    logger.info("Generating diagnostic plots...")
    return plot_diagnostics_node(state)


def auto_query_node(state: VibraDiagMainState) -> dict[str, Any]:
    """
    Constructs an automated diagnostic query when only a signal file was provided.
    """
    signal_data = dict(state.get("signal_processing_result") or state.get("signal_analysis") or {})
    if "machine_metadata" not in signal_data and state.get("machine_metadata"):
        signal_data["machine_metadata"] = state.get("machine_metadata")
    if "time_domain_stats" not in signal_data and state.get("time_domain_stats"):
        signal_data["time_domain_stats"] = state.get("time_domain_stats")
    if "primary_fault" not in signal_data and state.get("primary_fault"):
        signal_data["primary_fault"] = state.get("primary_fault")

    auto_query = build_initial_signal_query(signal_data)
    logger.info("AutoQuery generated query: '%s'", auto_query[:100])
    return {
        "user_query": auto_query,
    }


def retrieval_node(
    state: VibraDiagMainState,
    retrieval_graph: RetrievalGraph | None = None,
) -> dict[str, Any]:
    """
    Executes RetrievalGraph, filtering excluded chunks and assembling merged context.
    """
    search_query = state.get("reformulated_query") or state.get("user_query", "").strip()
    excluded_ids = set(state.get("excluded_chunk_ids") or [])

    if not search_query:
        logger.warning("retrieval_node called with empty search_query.")
        return {
            "text_passages": [],
            "visual_evidence": [],
            "merged_context": "",
        }

    graph = retrieval_graph or get_default_retrieval_graph()
    logger.info("Executing RetrievalGraph for query: '%s' (excluded=%d)", search_query[:60], len(excluded_ids))

    try:
        ret_result = graph.invoke(query=search_query, deduplicate_parents=True)
        raw_text_passages = ret_result.get("text_passages", [])
        raw_visual_evidence = ret_result.get("visual_evidence", [])

        filtered_passages = [
            p for p in raw_text_passages if p.get("id") not in excluded_ids
        ]
        filtered_visual = [
            v for v in raw_visual_evidence if v.get("id") not in excluded_ids
        ]

        merged_context = "\n\n".join(
            [f"[Kaynak: {p.get('id', 'N/A')}]\n{p.get('text', '')}" for p in filtered_passages if p.get("text")]
        )

        return {
            "text_passages": filtered_passages,
            "visual_evidence": filtered_visual,
            "merged_context": merged_context,
            "sub_query_passages": ret_result.get("sub_query_passages"),
            "max_text_score": ret_result.get("max_text_score", 0.0),
        }
    except Exception as e:
        logger.error("retrieval_node execution failed: %s", e, exc_info=True)
        return {
            "errors": [f"Retrieval hatası: {e}"],
            "text_passages": [],
            "visual_evidence": [],
            "merged_context": "",
        }


def generation_node_wrapper(state: VibraDiagMainState) -> dict[str, Any]:
    """
    Invokes generation_node, ensuring ISO Zone D emergency guidance is incorporated.
    """
    state_dict = dict(state)

    signal_data = dict(state_dict.get("signal_processing_result") or state_dict.get("signal_analysis") or {})
    if signal_data:
        if "machine_metadata" not in signal_data and state_dict.get("machine_metadata"):
            signal_data["machine_metadata"] = state_dict.get("machine_metadata")
        if "time_domain_stats" not in signal_data and state_dict.get("time_domain_stats"):
            signal_data["time_domain_stats"] = state_dict.get("time_domain_stats")
        if "primary_fault" not in signal_data and state_dict.get("primary_fault"):
            signal_data["primary_fault"] = state_dict.get("primary_fault")
        state_dict["signal_processing_result"] = signal_data

    iso_zone = signal_data.get("iso_severity_zone") or signal_data.get("iso_zone")

    if iso_zone == "D" and not state_dict.get("previous_failure_reason"):
        logger.warning("ISO Zone D (Danger/Emergency) detected! Enforcing emergency directive.")

    logger.info("Executing generation_node...")
    gen_result = generation_node(state_dict)
    return gen_result


def self_corrector_node_wrapper(
    state: VibraDiagMainState,
    thresholds_override: dict[str, Any] | None = None,
    judge_client: Any | None = None,
    groq_client: Any | None = None,
) -> dict[str, Any]:
    """
    Invokes self_corrector_node to validate the candidate response and manage retries.
    """
    logger.info("Executing self_corrector_node...")
    corrector_output = self_corrector_node(
        state=state,
        thresholds_override=thresholds_override,
        judge_client=judge_client,
        groq_client=groq_client,
    )
    return corrector_output


def finalize_response_node(state: VibraDiagMainState) -> dict[str, Any]:
    """
    Finalizes response output, updates messages history with AIMessage, and sets completion status.
    """
    llm_response = state.get("llm_response", "").strip()
    errors = state.get("errors") or []

    if not llm_response and errors:
        llm_response = "İşlem sırasında bir hata oluştu:\n- " + "\n- ".join(errors)

    ai_msg = AIMessage(content=llm_response)

    logger.info(
        "FinalizeResponse: Response length=%d chars, is_flagged=%s, errors=%d",
        len(llm_response),
        state.get("is_flagged", False),
        len(errors),
    )

    return {
        "llm_response": llm_response,
        "messages": [ai_msg],
    }



def entry_router(state: VibraDiagMainState) -> Literal["signal_loader", "retrieval", "finalize_response"]:
    """Routes after input analysis."""
    decision = state.get("route_decision")
    if decision in ("signal_only", "hybrid"):
        return "signal_loader"
    elif decision == "rag_only":
        return "retrieval"
    else:
        return "finalize_response"


def route_after_loader(state: VibraDiagMainState) -> Literal["signal_dsp", "finalize_response"]:
    """Routes after signal loader execution."""
    errors = state.get("errors") or []
    if errors or state.get("route_decision") == "error":
        return "finalize_response"
    return "signal_dsp"


def route_after_dsp(state: VibraDiagMainState) -> Literal["auto_query", "retrieval", "finalize_response"]:
    """Routes after signal DSP and diagnostic plot creation."""
    errors = state.get("errors") or []
    if errors or state.get("route_decision") == "error":
        return "finalize_response"

    user_query = (state.get("user_query") or "").strip()
    if not user_query:
        return "auto_query"
    return "retrieval"


def route_after_corrector(
    state: VibraDiagMainState,
) -> Literal["retrieval", "generation", "finalize_response"]:
    """Routes based on Self-Corrector verification outcome."""
    decision = state.get("route_decision")
    if decision == "retry_retrieval":
        logger.info("Corrector decision: 'retry_retrieval' -> Routing back to retrieval_node.")
        return "retrieval"
    elif decision == "retry_generation":
        logger.info("Corrector decision: 'retry_generation' -> Routing back to generation_node.")
        return "generation"
    else:
        logger.info("Corrector decision: '%s' -> Proceeding to finalize_response.", decision)
        return "finalize_response"


def build_main_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    retrieval_graph: RetrievalGraph | None = None,
    corrector_thresholds: dict[str, Any] | None = None,
    corrector_judge_client: Any | None = None,
    corrector_groq_client: Any | None = None,
) -> Any:
    """
    Constructs and compiles the complete VibraDiag application LangGraph.

    Parameters
    ----------
    checkpointer : BaseCheckpointSaver, optional
        Checkpointer for conversation state persistence (e.g. MemorySaver()).
    retrieval_graph : RetrievalGraph, optional
        Custom or mocked RetrievalGraph instance (defaults to lazy singleton).
    corrector_thresholds : dict, optional
        Threshold overrides for self-corrector checks.
    corrector_judge_client : Any, optional
        Custom LLM judge client for evaluator checks.
    corrector_groq_client : Any, optional
        Custom Groq client for query rewriter.

    Returns
    -------
    CompiledStateGraph
        Executable compiled StateGraph instance.
    """
    workflow = StateGraph(VibraDiagMainState)

    workflow.add_node("input_router", input_router_node)
    workflow.add_node("signal_loader", signal_loader_node)
    workflow.add_node("signal_dsp", signal_dsp_node)
    workflow.add_node("diagnostic_plots", diagnostic_plots_wrapper_node)
    workflow.add_node("auto_query", auto_query_node)

    def _bound_retrieval_node(state: VibraDiagMainState) -> dict[str, Any]:
        return retrieval_node(state, retrieval_graph=retrieval_graph)

    def _bound_corrector_node(state: VibraDiagMainState) -> dict[str, Any]:
        return self_corrector_node_wrapper(
            state,
            thresholds_override=corrector_thresholds,
            judge_client=corrector_judge_client,
            groq_client=corrector_groq_client,
        )

    workflow.add_node("retrieval", _bound_retrieval_node)
    workflow.add_node("generation", generation_node_wrapper)
    workflow.add_node("self_corrector", _bound_corrector_node)
    workflow.add_node("finalize_response", finalize_response_node)

    workflow.add_edge(START, "input_router")

    workflow.add_conditional_edges(
        "input_router",
        entry_router,
        {
            "signal_loader": "signal_loader",
            "retrieval": "retrieval",
            "finalize_response": "finalize_response",
        },
    )

    workflow.add_conditional_edges(
        "signal_loader",
        route_after_loader,
        {
            "signal_dsp": "signal_dsp",
            "finalize_response": "finalize_response",
        },
    )

    workflow.add_edge("signal_dsp", "diagnostic_plots")

    workflow.add_conditional_edges(
        "diagnostic_plots",
        route_after_dsp,
        {
            "auto_query": "auto_query",
            "retrieval": "retrieval",
            "finalize_response": "finalize_response",
        },
    )

    workflow.add_edge("auto_query", "retrieval")
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", "self_corrector")

    workflow.add_conditional_edges(
        "self_corrector",
        route_after_corrector,
        {
            "retrieval": "retrieval",
            "generation": "generation",
            "finalize_response": "finalize_response",
        },
    )

    workflow.add_edge("finalize_response", END)

    active_checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    return workflow.compile(checkpointer=active_checkpointer)
