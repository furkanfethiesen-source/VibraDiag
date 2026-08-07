"""
Prompt Builder for VibraDiag Generation Layer.

Assembles system prompts (combining base guidelines with structured signal analysis context)
and formats user messages (combining retrieved RAG context with user query).
"""

from __future__ import annotations

import json
from typing import Any

from config_loader import load_prompts_cfg


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

    metadata = signal_data.get("metadata", {})
    if isinstance(metadata, dict):
        machine_name = metadata.get("machine_name", "")
        machine_type = metadata.get("machine_type", "")
        rpm = metadata.get("rpm", 0.0)
        point = metadata.get("measurement_point", "")
        if machine_name or machine_type or rpm:
            lines.append(
                f"- Makine Bilgisi: {machine_name} | Tip: {machine_type} | RPM: {rpm} | Ölçüm Noktası: {point}"
            )
    elif hasattr(metadata, "machine_name"):
        lines.append(
            f"- Makine Bilgisi: {metadata.machine_name} | Tip: {metadata.machine_type} | RPM: {metadata.rpm}"
        )

    rms = signal_data.get("overall_rms") or signal_data.get("vibration_velocity_rms_mm_s")
    peak = signal_data.get("peak_value")
    crest = signal_data.get("crest_factor")
    kurt = signal_data.get("kurtosis")
    iso_zone = signal_data.get("iso_zone") or signal_data.get("iso_severity_zone")
    iso_meaning = signal_data.get("iso_severity_meaning", "")

    stats = []
    if rms is not None:
        stats.append(f"RMS: {rms:.3f} mm/s")
    if peak is not None:
        stats.append(f"Peak: {peak:.3f}")
    if crest is not None:
        stats.append(f"Crest Factor: {crest:.2f}")
    if kurt is not None:
        stats.append(f"Kurtosis: {kurt:.2f}")
    if stats:
        lines.append(f"- Zaman Alanı İstatistikleri: {', '.join(stats)}")

    if iso_zone:
        lines.append(f"- ISO Değerlendirmesi: Bölge {iso_zone} ({iso_meaning})")

    dominant = signal_data.get("dominant_frequencies", [])
    if dominant:
        lines.append("- Baskın Frekans Bileşenleri:")
        for df in dominant:
            if isinstance(df, dict):
                lines.append(f"  * {df.get('order', 'N/A')}: {df.get('freq_hz', 0.0)} Hz (Genlik: {df.get('amplitude', 0.0)})")
            else:
                lines.append(f"  * {df}")

    fault_indicators = signal_data.get("fault_indicators", [])
    envelope_diag = signal_data.get("envelope_diagnosis", {})
    direct_diag = signal_data.get("direct_spectrum_diagnosis", {})
    recip_diag = signal_data.get("reciprocating_diagnosis", {})

    if fault_indicators:
        lines.append("- Tespit Edilen Arıza Göstergeleri:")
        for fi in fault_indicators:
            if isinstance(fi, dict):
                lines.append(
                    f"  * {fi.get('type', 'Bilinmeyen Arıza')}: "
                    f"Güven={fi.get('confidence', 0.0)}, Açıklama={fi.get('description', '')}"
                )

    if envelope_diag:
        lines.append("- Rulman Zarf Analizi Teşhisi:")
        lines.append(f"  {json.dumps(envelope_diag, ensure_ascii=False, indent=2)}")

    if direct_diag:
        lines.append("- Doğrudan Spektrum Analizi Teşhisi:")
        lines.append(f"  {json.dumps(direct_diag, ensure_ascii=False, indent=2)}")

    if recip_diag:
        lines.append("- Pistonlu Makine Teşhisi:")
        lines.append(f"  {json.dumps(recip_diag, ensure_ascii=False, indent=2)}")

    summary = signal_data.get("expert_summary", "")
    if summary:
        lines.append(f"- Uzman Ön Analiz Özeti:\n  {summary}")

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

    prompts_cfg = load_prompts_cfg()
    gen_prompt_cfg = getattr(prompts_cfg, "generation_prompt", {}) or {}
    template = gen_prompt_cfg.get(
        "initial_signal_query_template",
        "{machine_type} makinesinin {measurement_point} bölgesinde {fault_name} tespiti yapılmıştır.\n"
        "ISO Şiddet Değerlendirmesi: Bölge {iso_zone} ({iso_meaning}).\n"
        "Titreşim Metrikleri: RMS {rms} mm/s, Peak {peak}, Crest Factor {crest_factor}, Kurtosis {kurtosis}.\n\n"
        "Bu hatayı teknik olarak detaylıca açıkla, olası kök nedenleri sırala ve bu durum için yapılması gereken acil müdahale ile önleyici bakım adımlarını detaylandır."
    )

    metadata = signal_data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    machine_type = metadata.get("machine_type") or metadata.get("machine_name") or "Ekipman"
    measurement_point = metadata.get("measurement_point") or "Ölçüm Noktası"

    fault_indicators = signal_data.get("fault_indicators", [])
    fault_names = []
    for fi in fault_indicators:
        if isinstance(fi, dict) and fi.get("type"):
            fault_names.append(str(fi.get("type")))
        elif isinstance(fi, str):
            fault_names.append(fi)

    fault_name = ", ".join(fault_names) if fault_names else "mekanik arıza belirtisi"

    iso_zone = signal_data.get("iso_zone") or signal_data.get("iso_severity_zone") or "N/A"
    iso_meaning = signal_data.get("iso_severity_meaning") or "Belirtilmedi"

    rms = signal_data.get("overall_rms") or signal_data.get("vibration_velocity_rms_mm_s")
    rms_str = f"{rms:.3f}" if rms is not None else "N/A"

    peak = signal_data.get("peak_value")
    peak_str = f"{peak:.3f}" if peak is not None else "N/A"

    crest = signal_data.get("crest_factor")
    crest_str = f"{crest:.2f}" if crest is not None else "N/A"

    kurt = signal_data.get("kurtosis")
    kurt_str = f"{kurt:.2f}" if kurt is not None else "N/A"

    return template.format(
        machine_type=machine_type,
        measurement_point=measurement_point,
        fault_name=fault_name,
        iso_zone=iso_zone,
        iso_meaning=iso_meaning,
        rms=rms_str,
        peak=peak_str,
        crest_factor=crest_str,
        kurtosis=kurt_str,
    ).strip()


def build_system_prompt(
    has_signal_context: bool = False,
    state: dict[str, Any] | None = None,
    is_followup: bool = False,
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

    return "\n\n".join(prompt_parts).strip()


def build_user_message(query: str, context: str = "") -> str:
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

