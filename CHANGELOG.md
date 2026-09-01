# Changelog

This file is used to record significant changes throughout the project on a dated basis.

## [2026-07-29] — Project initialized
### Added
- Repository structure created.
- Initial commit made.

---

## [2026-07-30 — 2026-07-31] — Core data setup
### Added
- Config files and base data schemas added.
- Main data document and data processing pipeline set up (See [DECISIONS.md: Ingestion](DECISIONS.md#1-ingestion)).
- .toml file added for package management.

---

## [2026-08-04 — 2026-08-06] — Ingestion & retrieval pipeline draft
### Added
- DSP subgraph creation pipeline set up; ingestion output loaded as sample chunks (See [DECISIONS.md: DSP](DECISIONS.md#2-dsp-digital-signal-processing)).
- Vector DB integration and retrieval pipeline set up (See [DECISIONS.md: Retrieval](DECISIONS.md#3-retrieval)).
- Constant values and config files added for the retrieval module.

### Changed
- Parent chunk storage strategy changed from json to sqlite db (See [DECISIONS.md: Parent-Child Chunking Strategy](DECISIONS.md#31-retrieval-parent-child-chunking-strategy-qdrant--sqlite)).

---

## [2026-08-07] — Generation module
### Added
- Generation module set up (system prompt, state management, plot function) (See [DECISIONS.md: Generation](DECISIONS.md#4-generation)).

### Changed
- Improvements made to the generation prompt as signal data and the follow up guideline is added to the prompt.

---

## [2026-08-11] — Evaluation pipeline & retrieval improvements
### Added
- Evaluation pipeline set up; retrieval benchmark dataset added (See [DECISIONS.md: RAGAS + Deterministic Evaluation Metrics](DECISIONS.md#71-evaluation-ragas--deterministic-evaluation-metrics)).
- Deduplication and hybrid search parameters added to the retrieval subgraph (See [DECISIONS.md: Hybrid Search with Dynamic Prefetch](DECISIONS.md#32-qdrant-hybrid-search-with-dynamic-prefetch), [DECISIONS.md: Parent Chunk Deduplication Mechanism](DECISIONS.md#36-parent-chunk-deduplication-mechanism-toggleable)).
- Max retry control added.

### Changed
- Text parser updated to be primarily heading-focused (See [DECISIONS.md: Section Heading Detection](DECISIONS.md#12-ingestion-section-heading-detection-via-font-analysis), [DECISIONS.md: Hierarchical Section Path](DECISIONS.md#11-ingestion-hierarchical-section-path-622--62--6)).
- Subgraph nodes made async for optimization (See [DECISIONS.md: Asynchronous Retrieval Subgraph](DECISIONS.md#33-retrieval-subgraph-asynchronous-architecture)).

---

## [2026-08-13] — Query decomposer module
### Added
- Query decomposer module built and integrated into the retrieval graph (See [DECISIONS.md: Decomposer Router Logistic Regression Classifier](DECISIONS.md#51-decomposer-router-logistic-regression-classifier)).
- Separate folder created for decomposer training and test results.
- Dynamic top-k and fallback mechanism added for when the max sub-query limit is exceeded (See [DECISIONS.md: Dynamic Top-k Based on Sub-Query Count](DECISIONS.md#34-retrieval-dynamic-top-k-based-on-sub-query-count), [DECISIONS.md: Max Sub-Query Overflow Protection](DECISIONS.md#52-decomposer-max-sub-query-overflow-protection)).

---

## [2026-08-16] — Self-corrector module
### Added
- Self-corrector module built (checker prompts, schema additions) (See [DECISIONS.md: Self-Corrector](DECISIONS.md#6-self-corrector)).
- Corrector evaluation system and example outputs added.
- Self-corrector integrated into the retriever.

### Fixed
- Clear error message added for missing API key.

---

## [2026-08-18 — 2026-08-19] — Main graph integration & model migration
### Added
- Script for building the main graph added; state enriched with loaded signal, primary fault, and checker results (See [DECISIONS.md: Automatic Initial Query](DECISIONS.md#43-generation-automatic-initial-query-build_initial_signal_query)).
- Self-corrector given the ability to pick the primary fault, strengthening the generation/retrieval error distinction (See [DECISIONS.md: Fault Consolidation via pick_primary_fault](DECISIONS.md#28-dsp-fault-consolidation-via-pick_primary_fault-and-weak-candidate-notification)).
- Sample outputs now generated as both JSON and kurtogram heatmap HTML.

### Changed
- Generation model migrated from Llama 70B to gpt-oss-120b due to Llama 70B deprecation (See [DECISIONS.md: Primary LLM Generator Model Selection](DECISIONS.md#41-generation-primary-llm-generator-model-selection-openaigpt-oss-120b-via-groq)).
- Answer extraction from the thinking process and prompt builder token cost optimized.
- Technical regex pattern improved.

---

## [2026-08-21 — 2026-08-22] — Fallback mechanisms & diagnostic enhancements
### Added
- Required parameters added for the Streamlit, API, and config modules.
- Soft fallback mechanisms added to the reranker and decomposer; retriever integration completed (See [DECISIONS.md: Reranker Soft-Fallback](DECISIONS.md#35-reranker-gradual-soft-fallback-mechanism), [DECISIONS.md: Decomposer Overflow Protection](DECISIONS.md#52-decomposer-max-sub-query-overflow-protection)).
- Chunk exclusion support added via domain dictionary expansion.
- Orbit/phase plot and envelope spectrum charts added.
- First evaluation report uploaded.

### Changed
- Dual LLM calls reduced to a single call, optimizing latency and token cost (See [DECISIONS.md: Reducing Dual LLM Calls to a Single Call](DECISIONS.md#63-self-corrector-reducing-dual-llm-calls-to-a-single-call-and-threshold-calibration)).
- Threshold calibration performed to fix the self-corrector being overly strict (See [DECISIONS.md: Self-Corrector Threshold Calibration](DECISIONS.md#63-self-corrector-reducing-dual-llm-calls-to-a-single-call-and-threshold-calibration)).
- Diagnosis mechanism made dual-validated using peak + RMS energy instead of just focusing on one (See [DECISIONS.md: Hybrid 2-Stage Peak Matching](DECISIONS.md#22-dsp-hybrid-2-stage-peak-matching-rms--peak-gating)).

---

## [2026-08-23] — Streamlit UI
### Added
- Streamlit UI module added.

---

## [2026-08-28] — Documentation
### Added
- README updated; DECISIONS file added explaining decisions made along with their rationale and outcomes (See [DECISIONS.md](DECISIONS.md)).

---

## [2026-08-29 — 2026-08-30] — Tracing, unified evaluation & stability fixes
### Added
- Support for loading via .env added.
- Langsmith tracing added to generation and main graph components.
- Unified judge prompt extension added to the main evaluation module, along with corrector add/remove support (See [DECISIONS.md: Unified Judge Checker](DECISIONS.md#63-self-corrector-reducing-dual-llm-calls-to-a-single-call-and-threshold-calibration)).
- Gemini fallback added for production errors in the main model (gpt-oss-120b) (See [DECISIONS.md: Exponential Backoff and Gemini Fallback](DECISIONS.md#42-generation-exponential-backoff-and-gemini-fallback)).

### Fixed
- Incorrect inconsistency flagging in the ISO zone consistency check fixed by loosening the scope.
- Minor inconsistencies fixed in the README.md .
### Notes
- Project is in active development; the agentic RAG + DSP-based bearing fault diagnosis system continues to mature.

---

## [2026-09-01] — Prompt reconstruction and DSP optimization
### Added
- Written a benchmark script that evaluates and records 20 labeled signal test results (See [DECISIONS.md: Fault Consolidation via pick_primary_fault](DECISIONS.md#28-dsp-fault-consolidation-via-pick_primary_fault-and-weak-candidate-notification)).
- Because of dominance of modulated sidebands in more severe damaged situations, dynamic SFER and compound fault arbitration were added to resolve secondary/co-occurring bearing defects when needed (See [DECISIONS.md: Dynamic SFER for Advanced Flaw Degradation](DECISIONS.md#29-dsp-dynamic-sideband-family-energy-ratio-sfer-for-advanced-flaw-degradation), [DECISIONS.md: Compound Defect Resolution & Multi-Fault Hierarchy](DECISIONS.md#211-dsp--architecture-compound-defect-resolution--multi-fault-hierarchy)).
- Core prompt reconstructed into "Anchor & Action" dual-tier architecture to strictly ground theoretical fault signatures while allowing disciplined maintenance action guidance (See [DECISIONS.md: "Anchor & Action" Prompt Architecture & Qualitative Grounding](DECISIONS.md#45-generation-anchor--action-prompt-architecture--qualitative-grounding)).
- Formal "Controlled Parametric Extrapolation" methodology note attached to evaluation deliverables (See [DECISIONS.md: Controlled Parametric Extrapolation & MVP Methodology Disclosure](DECISIONS.md#72-evaluation-controlled-parametric-extrapolation--mvp-methodology-disclosure)).

### Changed
- Generator output token budget optimized from 4096 to 1500 max tokens to eliminate Groq 8000 TPM HTTP 413 rate limit ceiling issues (See [DECISIONS.md: Output Token Budget Optimization](DECISIONS.md#46-generation-output-token-budget-optimization-1500-max-tokens--tpm-ceiling-protection)).

### Fixed
- Key optimizations implemented in DSP pipeline including physical channel energy weighting and power cepstrum quefrency spacing analysis (See [DECISIONS.md: Physical Channel Energy Weighting](DECISIONS.md#210-dsp-physical-channel-energy-weighting-in-multi-channel-fault-arbitration), [DECISIONS.md: Power Cepstrum Quefrency Analysis](DECISIONS.md#212-dsp-power-cepstrum-quefrency-analysis-for-sideband-harmonic-spacing)).