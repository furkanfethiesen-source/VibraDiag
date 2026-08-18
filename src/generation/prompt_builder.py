"""
Prompt Builder for VibraDiag Generation Layer.

Assembles system prompts (combining base guidelines with structured signal analysis context)
and formats user messages (combining retrieved RAG context with user query).
"""

from __future__ import annotations

import json
from typing import Any

from config_loader import load_prompts_cfg


from deterministic_tools.fault_analyzer import pick_primary_fault


def build_signal_context(signal_data: dict[str, Any] | None) -> str:
    """
    Format structured signal analysis results into a clean markdown text representation for the LLM.

    Parameters
    ----------
    signal_data : dict or None
        Signal analysis dictionary (e.g. SignalAnalysisResult or signal_processing_result dict).

    Returns
    -------
    str
        Formatted markdown representation of signal findings.
    """
    if not signal_data:
        return ""

    lines = ["=== SİNYAL ANALİZİ VE ARİZA BULGULARI ==="]

    metadata = signal_data.get("machine_metadata") or signal_data.get("metadata") or {}
    if isinstance(metadata, dict):
        machine_name = metadata.get("machine_name", "")
        machine_type = metadata.get("machine_type", "")
        rpm = metadata.get("rpm", 0.0)
        point = metadata.get("measurement_point", "")
        if machine_name or machine_type or rpm:
            lines.append(
                f"- Makine Bilgisi: {machine_name or 'Endüstriyel Ekipman'} | Tip: {machine_type} | RPM: {rpm}"
                + (f" | Ölçüm Noktası: {point}" if point else "")
            )
    elif hasattr(metadata, "machine_name"):
        lines.append(
            f"- Makine Bilgisi: {metadata.machine_name} | Tip: {metadata.machine_type} | RPM: {metadata.rpm}"
        )

    td_stats = signal_data.get("time_domain_stats") or {}
    rms = td_stats.get("overall_rms") or signal_data.get("overall_rms") or signal_data.get("vibration_velocity_rms_mm_s")
    peak = td_stats.get("peak_value") or signal_data.get("peak_value")
    crest = td_stats.get("crest_factor") or signal_data.get("crest_factor")
    kurt = td_stats.get("kurtosis") or signal_data.get("kurtosis")
    iso_zone = signal_data.get("iso_severity_zone") or signal_data.get("iso_zone")
    iso_meaning = signal_data.get("iso_severity_meaning", "")

    stats = []
    if rms is not None:
        stats.append(f"RMS: {float(rms):.3f} mm/s")
    if peak is not None:
        stats.append(f"Peak: {float(peak):.3f}")
    if crest is not None:
        stats.append(f"Crest Factor: {float(crest):.2f}")
    if kurt is not None:
        stats.append(f"Kurtosis: {float(kurt):.2f}")
    if stats:
        lines.append(f"- Zaman Alanı İstatistikleri: {', '.join(stats)}")

    if iso_zone:
        lines.append(f"- ISO Değerlendirmesi: Bölge {iso_zone} ({iso_meaning})")

    fault_summary = pick_primary_fault(
        direct_spectrum_diagnosis=signal_data.get("direct_spectrum_diagnosis"),
        envelope_diagnosis=signal_data.get("envelope_diagnosis"),
        reciprocating_diagnosis=signal_data.get("reciprocating_diagnosis"),
    )
    primary_fault = fault_summary.get("primary_fault_name")
    if primary_fault:
        lines.append(
            f"- Birincil Tespit Edilen Arıza: **{primary_fault}** "
            f"(Güven: {fault_summary.get('confidence', 0.0):.2f}, Şiddet Skoru: {fault_summary.get('score', 0.0):.2f})"
        )

    all_faults = fault_summary.get("all_faults", [])
    if all_faults:
        lines.append("- Doğrulanmış Birincil/İkincil Arıza Göstergeleri:")
        for f in all_faults:
            lines.append(
                f"  * **{f.get('name')}**: Güven={f.get('confidence', 0.0):.2f}, "
                f"Şiddet={f.get('severity', 0.0):.2f} (Skor={f.get('score', 0.0):.2f})"
            )

    weak_faults = fault_summary.get("weak_candidates", [])
    if weak_faults:
        lines.append("- Olası Düşük Frekanslı / Zayıf Göstergeler (Gürültü Şüphesi Taşıyan İkincil Belirtiler):")
        for wf in weak_faults[:4]:
            lines.append(
                f"  * {wf.get('name')}: Güven={wf.get('confidence', 0.0):.2f}, "
                f"Şiddet={wf.get('severity', 0.0):.2f} (Not: Yetersiz harmonik/yan bant nedeniyle zayıf adaydır)"
            )

    envelope_diag = signal_data.get("envelope_diagnosis", {})
    direct_diag = signal_data.get("direct_spectrum_diagnosis", {})
    recip_diag = signal_data.get("reciprocating_diagnosis", {})

    if envelope_diag:
        lines.append("- Rulman Zarf Analizi Teşhisi:")
        lines.append(f"  {json.dumps(envelope_diag, ensure_ascii=False, indent=2)}")

    if direct_diag:
        lines.append("- Doğrudan Spektrum Analizi Teşhisi:")
        lines.append(f"  {json.dumps(direct_diag, ensure_ascii=False, indent=2)}")

    if recip_diag:
        lines.append("- Pistonlu Makine Teşhisi:")
        lines.append(f"  {json.dumps(recip_diag, ensure_ascii=False, indent=2)}")

    return "\n".join(lines)


def build_initial_signal_query(signal_data: dict[str, Any] | None) -> str:
    """
    Construct a standardized initial diagnostic query from signal analysis output data.

    Parameters
    ----------
    signal_data : dict or None
        Signal analysis result dictionary.

    Returns
    -------
    str
        Standardized initial user query prompt string.
    """
    if not signal_data:
        return "Titreşim analizi sonuçlarını detaylıca açıkla ve bu durum için yapılması gerekenler nelerdir?"

    if isinstance(signal_data, object) and hasattr(signal_data, "model_dump"):
        signal_data = signal_data.model_dump()

    metadata = signal_data.get("machine_metadata") or signal_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    machine_name = metadata.get("machine_name") or metadata.get("machine_type") or "Ekipman"
    measurement_point = metadata.get("measurement_point") or "Ölçüm Noktası"

    fault_summary = pick_primary_fault(
        direct_spectrum_diagnosis=signal_data.get("direct_spectrum_diagnosis"),
        envelope_diagnosis=signal_data.get("envelope_diagnosis"),
        reciprocating_diagnosis=signal_data.get("reciprocating_diagnosis"),
    )
    fault_name = fault_summary.get("primary_fault_name") or "mekanik arıza belirtisi"

    iso_zone = signal_data.get("iso_severity_zone") or signal_data.get("iso_zone") or "N/A"
    iso_meaning = signal_data.get("iso_severity_meaning") or "Belirtilmedi"

    td_stats = signal_data.get("time_domain_stats") or {}
    rms = td_stats.get("overall_rms") or signal_data.get("overall_rms") or signal_data.get("vibration_velocity_rms_mm_s")
    rms_str = f"{float(rms):.3f}" if rms is not None else "N/A"

    peak = td_stats.get("peak_value") or signal_data.get("peak_value")
    peak_str = f"{float(peak):.3f}" if peak is not None else "N/A"

    crest = td_stats.get("crest_factor") or signal_data.get("crest_factor")
    crest_str = f"{float(crest):.2f}" if crest is not None else "N/A"

    kurt = td_stats.get("kurtosis") or signal_data.get("kurtosis")
    kurt_str = f"{float(kurt):.2f}" if kurt is not None else "N/A"

    query_text = (
        f"{machine_name} makinesinin {measurement_point} bölgesinde {fault_name} tespiti yapılmıştır.\n"
        f"ISO Şiddet Değerlendirmesi: Bölge {iso_zone} ({iso_meaning}).\n"
        f"Titreşim Metrikleri: RMS {rms_str} mm/s, Peak {peak_str}, Crest Factor {crest_str}, Kurtosis {kurt_str}.\n\n"
        "Bu hatayı teknik olarak detaylıca açıkla, olası kök nedenleri sırala ve bu durum için yapılması gereken acil müdahale ile önleyici bakım adımlarını detaylandır."
    )
    return query_text


def build_system_prompt(
    has_signal_context: bool = False,
    state: dict[str, Any] | None = None,
    is_followup: bool = False,
    is_decomposed: bool = False,
) -> str:
    """
    Construct system prompt combining base prompt and optional signal context.

    Parameters
    ----------
    has_signal_context : bool
        Whether signal analysis data is available.
    state : dict, optional
        Pipeline state containing signal analysis data.
    is_followup : bool
        Whether this is a follow-up turn in an ongoing chat.

    Returns
    -------
    str
        Full system prompt string.
    """
    prompts_cfg = load_prompts_cfg()
    gen_prompt_cfg = getattr(prompts_cfg, "generation_prompt", {}) or {}

    base_prompt = gen_prompt_cfg.get(
        "base",
        "Sen VibraDiag, endüstriyel titreşim analizi ve mekanik arıza teşhisi konusunda uzmanlaşmış bir yapay zeka asistanısın.",
    )

    prompt_parts = [base_prompt.strip()]

    if is_followup:
        followup_guidelines = gen_prompt_cfg.get("followup_guidelines", "")
        if followup_guidelines:
            prompt_parts.append(f"\n=== TAKİP SORUSU YÖNERGESİ ===\n{followup_guidelines.strip()}")

    if has_signal_context and state:
        signal_prefix = gen_prompt_cfg.get(
            "signal_context_prefix",
            "\n\n=== CANLI SİNYAL ANALİZ SONUÇLARI ===\nCevabını oluştururken aşağıdaki canlı analiz sonuçlarını da değerlendir.\n",
        )

        signal_data = (
            state.get("signal_analysis")
            or state.get("signal_processing_result")
        )

        if isinstance(signal_data, object) and hasattr(signal_data, "model_dump"):
            signal_data = signal_data.model_dump()

        formatted_signal = build_signal_context(signal_data)
        if formatted_signal:
            prompt_parts.append(f"\n{signal_prefix.strip()}\n\n{formatted_signal}")

    if is_decomposed:
        prompt_parts.append(f"\n{build_synthesis_directive()}")

    return "\n\n".join(prompt_parts).strip()


def build_user_message(
    query: str,
    context: str = "",
    sub_query_passages: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """
    Construct user message combining query and retrieved RAG context.

    Parameters
    ----------
    query : str
        User text query.
    context : str
        Merged document context from retrieval phase.

    Returns
    -------
    str
        Formatted user prompt.
    """
    clean_query = query.strip()
    clean_context = context.strip()

    if sub_query_passages:
        decomposed_context, _ = build_decomposed_context(sub_query_passages)
        return (
            f"{decomposed_context}\n\n"
            f"=== ORİJİNAL SORU ===\n"
            f"{clean_query}"
        )

    if not clean_context:
        return clean_query

    return (
        f"=== DOKÜMANTASYON BAĞLAMI ===\n"
        f"{clean_context}\n\n"
        f"=== KULLANICI SORUSU ===\n"
        f"{clean_query}"
    )


def build_messages_payload(
    system_prompt: str,
    user_message: str,
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Assemble complete OpenAI/Groq compliant messages payload.

    Parameters
    ----------
    system_prompt : str
        The active system prompt.
    user_message : str
        The latest user prompt (with documentation context embedded if available).
    chat_history : list of dict, optional
        Previous turns in the conversation [{"role": "user"|"assistant", "content": "..."}, ...].

    Returns
    -------
    list of dict
        Full message payload.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for turn in chat_history:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                messages.append({"role": str(turn["role"]), "content": str(turn["content"])})
    messages.append({"role": "user", "content": user_message})
    return messages


def build_decomposed_context(
    sub_query_passages: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, str]]:
    """
    Sub-query bazlı gruplu context oluşturur, citation etiketleri yerleştirir.

    Parameters
    ----------
    sub_query_passages : dict
        {sub_query_text: [passage_dicts]} mapping'i.

    Returns
    -------
    tuple[str, dict[str, str]]
        (formatted_context, citation_map)
        citation_map: {"K1": "source_description", ...}
    """
    lines = ["=== ARAŞTIRMA BAĞLAMI ==="]
    citation_map: dict[str, str] = {}
    citation_counter = 1

    for sq_idx, (sub_query, passages) in enumerate(sub_query_passages.items(), 1):
        lines.append(f"\n[Alt Sorgu {sq_idx}: \"{sub_query}\"]")

        if not passages:
            lines.append("  (Bu alt sorgu için sonuç bulunamadı)")
            continue

        for passage in passages:
            cite_key = f"K{citation_counter}"
            chunk_id = passage.get("id", "unknown")
            metadata = passage.get("metadata", {})
            page_number = metadata.get("page_number", "?")
            section_path = metadata.get("section_path", "")
            text = passage.get("text", "")

            source_desc = f"{section_path}, s.{page_number}" if section_path else f"chunk:{chunk_id}, s.{page_number}"
            citation_map[cite_key] = source_desc

            lines.append(f"  [{cite_key}] {source_desc}:")
            lines.append(f"    {text[:500]}{'...' if len(text) > 500 else ''}")

            citation_counter += 1

    return "\n".join(lines), citation_map


def build_synthesis_directive(citation_map: dict[str, str] | None = None) -> str:
    """
    Generator system prompt'una eklenen citation zorunluluğu direktifi.

    Parameters
    ----------
    citation_map : dict, optional
        Mevcut citation haritası (logging/debug için).

    Returns
    -------
    str
        Citation directive metni.
    """
    prompts_cfg = load_prompts_cfg()
    decomposer_prompt_cfg = getattr(prompts_cfg, "decomposer_prompt", {}) or {}

    directive = decomposer_prompt_cfg.get(
        "citation_directive",
        "\n=== KAYNAK ZORUNLULUĞU ===\n"
        "Her teknik iddia için köşeli parantez içinde kaynak numarasını belirt: [K1], [K2].\n"
        "Birden fazla kaynaktan destek alınıyorsa: [K1, K3].\n"
        "Kaynak gösterilemeyen bir iddia kesinlikle yazılmamalıdır.\n"
        "Yanıtın sonunda 'Kullanılan Kaynaklar:' başlığı altında kaynak listesi ver.",
    )
    return directive.strip()

