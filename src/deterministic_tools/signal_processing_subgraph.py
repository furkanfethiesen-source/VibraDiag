
"""
VibraDiag - Signal Processing SubGraph ve Iki Seviyeli State (Faz 6)
=========================================================================

ONEMLI ON-KOSUL (kullaniciyla konusuldu): ana LangGraph (VibraDiag'in tum
agent'i — RAG, konusma, diger alt-graflar) HENUZ TASARLANMADI. Bu dosya
o yuzden ikiye ayriliyor:

KESIN / su an test edilebilir (ana graftan bagimsiz):
    - SignalProcessingSubGraph (Adim 6.1) — Faz 4/5'teki tum node'lari
    (kurtogram, fallback, rotating/reciprocating expert, iso_check)
    TEK bir derlenmis alt-grafta toplar.
    - SignalProcessingInternalState (Adim 6.2'nin yerel yarisi) —
    machine_expert_nodes.py'de tanimli, degismedi.

TASLAK / ana graf tasarlanana kadar degisebilir:
    - VibraDiagMainState (Adim 6.2'nin global yarisi) — SADECE bu
    subgraph'in parent'tan bekledigi/parent'a dondurdugu alanlari
    icerir. Ana grafin gercek sekli belli olunca genisletilecek
    (TypedDict'ler additive oldugu icin bu guvenli).
    - map_parent_to_subgraph / map_subgraph_to_parent (Adim 6.3) —
    arayuz sozlesmesi olarak hazir, ama gercek parent graf onlari
    cagirana kadar sadece bu dosyanin __main__ testinde dogrulaniyor.
    - iso_zone_router (Adim 6.4) — fonksiyon hazir ('D' zone'da
    "emergency" donuyor), ama HENUZ HICBIR CONDITIONAL EDGE'E
    BAGLI DEGIL cunku ana grafta bir 'emergency' node'u yok. Ana
    graf yazildiginda tek satirla baglanabilir (asagida ornek).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from deterministic_tools.reader import LoadedSignal
from deterministic_tools.ıso_evaluator import iso_evaluator_node
from deterministic_tools.machine_expert_nodes import (
    SignalProcessingInternalState,
    fallback_band_node,
    kurtogram_node,
    kurtogram_reliability_router,
    machine_type_router,
    reciprocating_expert_node,
    rotating_expert_node,
)


def iso_check_node(state: SignalProcessingInternalState) -> dict:
    """iso_evaluator_node'un korumali (guarded) hali: ISO siniflandirmasi
    icin gereken alanlar (machine_class, vibration_velocity_rms_mm_s)
    state'te yoksa HATA VERMEZ, sessizce atlar. Boylece bu alt-graf, RMS
    hiz degeri hesaplanmamis cagrilarda da (orn. sadece rulman ariza
    teshisi isteniyorsa) sorunsuz calisir — ISO degerlendirmesi opsiyonel
    bir ek katman, zorunlu bir on-kosul degil."""
    if "vibration_velocity_rms_mm_s" not in state or "machine_class" not in state:
        return {}
    return iso_evaluator_node(state)


def build_signal_processing_subgraph():
    """Faz 4/5'teki tum node'lari TEK bir derlenmis alt-grafta toplar:

        START -> iso_check -> [machine_type_router]
                                "rotating" -> kurtogram -> [reliability?]
                                                    -> rotating_expert -> END
                                                    -> fallback_band -> rotating_expert -> END
                                "reciprocating" -> reciprocating_expert -> END

    iso_check her zaman ilk calisan node (ISO siniflandirmasi rulman/
    piston ayrimindan bagimsiz, genel titresim seviyesi degerlendirmesi
    oldugu icin routing'den ONCE mantikli)."""
    graph = StateGraph(SignalProcessingInternalState)

    graph.add_node("iso_check", iso_check_node)
    graph.add_node("kurtogram", kurtogram_node)
    graph.add_node("fallback_band", fallback_band_node)
    graph.add_node("rotating_expert", rotating_expert_node)
    graph.add_node("reciprocating_expert", reciprocating_expert_node)

    graph.set_entry_point("iso_check")
    graph.add_conditional_edges(
        "iso_check", machine_type_router,
        {"rotating": "kurtogram", "reciprocating": "reciprocating_expert"},
    )
    graph.add_conditional_edges(
        "kurtogram", kurtogram_reliability_router,
        {"rotating_expert": "rotating_expert", "fallback_band": "fallback_band"},
    )
    graph.add_edge("fallback_band", "rotating_expert")
    graph.add_edge("rotating_expert", END)
    graph.add_edge("reciprocating_expert", END)

    return graph.compile()


_COMPILED_SUBGRAPH = None


def _get_compiled_subgraph():
    global _COMPILED_SUBGRAPH
    if _COMPILED_SUBGRAPH is None:
        _COMPILED_SUBGRAPH = build_signal_processing_subgraph()
    return _COMPILED_SUBGRAPH



class VibraDiagMainState(TypedDict, total=False):
    """TASLAK global state. Ana graf (RAG/agent/konusma) henuz
    tasarlanmadigi icin bu SADECE SignalProcessingSubGraph'in parent'tan
    bekledigi girdi alanlarini ve parent'a dondurdugu TEK bir cikti
    alanini (signal_processing_result) icerir.

    Cikti TEK bir nested dict altinda toplandi (flatten edilmedi) —
    boylece ana grafa baska alt-graflar (RAG, vs.) eklendiginde alan adi
    catismasi riski sifira iner. Girdi alanlari ise dogrudan isimleriyle
    tutuluyor (bunlar zaten spesifik/az-catisma-riskli isimler: rpm,
    machine_type, vb.); ana graf tasarlanirken bu bolum degisebilir.
    """

    machine_type: Literal["rotating", "reciprocating"]
    loaded_signal: LoadedSignal
    rpm: float
    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle_deg: float
    n_cylinders: int
    stroke_type: str
    machine_class: str
    vibration_velocity_rms_mm_s: float

    signal_processing_result: dict


_SUBGRAPH_INPUT_KEYS = (
    "machine_type", "loaded_signal", "rpm", "n_balls", "ball_diameter",
    "pitch_diameter", "contact_angle_deg", "n_cylinders", "stroke_type",
    "machine_class", "vibration_velocity_rms_mm_s",
)



def map_parent_to_subgraph(parent_state: VibraDiagMainState) -> SignalProcessingInternalState:
    """VibraDiagMainState -> SignalProcessingInternalState.

    Sadece subgraph'in taniyacagi alanlari secip aktarir; parent state'te
    olabilecek alakasiz alanlar (RAG baglami, konusma gecmisi, vb.)
    subgraph'e SIZMAZ — state kirliligini onlemenin amaci bu."""
    return {k: parent_state[k] for k in _SUBGRAPH_INPUT_KEYS if k in parent_state}


def map_subgraph_to_parent(subgraph_output: dict) -> dict:
    """SignalProcessingInternalState (subgraph ciktisi) -> VibraDiagMainState
    guncellemesi. TEK bir nested alana ('signal_processing_result')
    toplanir (bkz. VibraDiagMainState docstring'i)."""
    return {"signal_processing_result": dict(subgraph_output)}


def signal_processing_subgraph_node(parent_state: VibraDiagMainState) -> dict:
    """Ana grafa TEK BIR NODE olarak eklenecek fonksiyon — Adim 6.1-6.3'un
    somut, kullanima hazir ciktisi. Ana graf ne icerirse icersin (RAG,
    baska agent'lar, vb.), bu fonksiyon DEGISMEDEN su sekilde eklenir:

        main_graph.add_node("signal_processing", signal_processing_subgraph_node)

    Ic islem: parent state'i mapper ile subgraph girdisine cevirir,
    derlenmis SignalProcessingSubGraph'i calistirir, sonucu tekrar
    mapper ile parent state guncellemesine cevirir."""
    subgraph_input = map_parent_to_subgraph(parent_state)
    subgraph = _get_compiled_subgraph()
    subgraph_output = subgraph.invoke(subgraph_input)
    return map_subgraph_to_parent(subgraph_output)


def iso_zone_router(parent_state: VibraDiagMainState) -> str:
    """Ana grafta 'signal_processing' node'undan SONRA conditional edge
    olarak kullanilmak uzere hazirlandi. iso_severity_zone == 'D'
    ("Not Permissible") ise "emergency" dondurur, aksi halde "continue".

    HENUZ HICBIR YERE BAGLI DEGIL: ana grafin bir 'emergency' node'u
    olmadigi icin (ana graf henuz tasarlanmadi) bu fonksiyon su an
    sadece bagimsiz cagrilabilir/test edilebilir durumda. Ana graf hazir
    oldugunda tek satirla baglanir:

        main_graph.add_conditional_edges(
            "signal_processing", iso_zone_router,
            {"emergency": "<senin_acil_durum_node_adin>", "continue": "<sonraki_node>"},
        )

    NOT: iso_severity_zone hesaplanmamissa (ISO girdileri saglanmadiysa,
    bkz. iso_check_node) "continue" doner — ISO degerlendirmesi olmadan
    "emergency" gibi kritik bir dala GIRMEK icin acik ve pozitif bir 'D'
    sonucu gerekir; belirsizlikte varsayilan GUVENLI tarafta (continue)
    kalinir, ama bu ayni zamanda 'D' teshisi kacirilmis olabilecegi
    anlamina da gelir — ISO girdileri saglanmadan bu router'a
    guvenilmemeli.
    """
    result = parent_state.get("signal_processing_result", {})
    zone = result.get("iso_severity_zone")
    return "emergency" if zone == "D" else "continue"
