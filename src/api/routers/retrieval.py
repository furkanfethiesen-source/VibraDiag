"""
VibraDiag API — Knowledge Base & Hybrid Retrieval Router.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends

from query_decomposer.classifier import QueryComplexityClassifier
from query_decomposer.llm_decomposer import LLMDecomposer
from retrieval.retriever import RetrievalGraph
from src.api.dependencies import get_retrieval_graph
from src.api.schemas.requests import DecomposeRequest, RetrievalSearchRequest
from src.api.schemas.responses import (
    DecomposeResponse,
    PassageDTO,
    RetrievalResponse,
    VisualEvidenceDTO,
)

router = APIRouter(prefix="/api/v1/retrieval", tags=["Knowledge Base & Retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="Hybrid Vector & DocStore Search",
    description="Performs dense + sparse BM42 vector search, cross-encoder reranking, and parent document resolution.",
)
async def search_knowledge_base(
    request: RetrievalSearchRequest,
    retrieval_graph: Annotated[RetrievalGraph, Depends(get_retrieval_graph)],
) -> RetrievalResponse:
    state_result = await retrieval_graph.ainvoke(
        query=request.query,
        text_top_k=request.text_top_k,
        visual_top_k=request.visual_top_k,
        threshold=request.threshold,
        deduplicate_parents=request.deduplicate_parents,
    )

    passages = []
    for p in state_result.get("text_passages", []):
        passages.append(
            PassageDTO(
                id=str(p.get("id", "N/A")),
                text=str(p.get("text", "")),
                score=float(p.get("score", 0.0)),
                metadata=p.get("metadata") or {},
            )
        )

    visual = []
    for v in state_result.get("visual_evidence", []):
        visual.append(
            VisualEvidenceDTO(
                id=str(v.get("id", "N/A")),
                text=str(v.get("text", "")),
                score=float(v.get("score", 0.0)),
                metadata=v.get("metadata") or {},
            )
        )

    merged_context = "\n\n".join(
        [f"[Kaynak: {p.id}]\n{p.text}" for p in passages if p.text]
    )

    return RetrievalResponse(
        query=request.query,
        is_complex=bool(state_result.get("is_complex", False)),
        sub_queries=state_result.get("sub_queries") or [],
        max_text_score=float(state_result.get("max_text_score", 0.0)),
        visual_triggered=bool(state_result.get("visual_triggered", False)),
        text_passages=passages,
        visual_evidence=visual,
        merged_context=merged_context,
    )


@router.post(
    "/decompose",
    response_model=DecomposeResponse,
    summary="Query Complexity & Decomposition Test",
    description="Evaluates whether a query requires decomposition and generates sub-queries via Groq LLM.",
)
async def decompose_query(
    request: DecomposeRequest,
) -> DecomposeResponse:
    classifier = QueryComplexityClassifier()
    clf_res = classifier.predict(request.query)

    sub_queries = []
    reasoning = None

    if clf_res.is_complex:
        decomposer = LLMDecomposer()
        dec_res = await asyncio.to_thread(decomposer.decompose, request.query)
        sub_queries = dec_res.sub_queries
        reasoning = dec_res.reasoning

    return DecomposeResponse(
        query=request.query,
        is_complex=clf_res.is_complex,
        classifier_confidence=clf_res.confidence,
        sub_queries=sub_queries,
        reasoning=reasoning,
    )
