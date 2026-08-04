
"""
VibraDiag - ISO Evaluator Node (Faz 5, Adim 5.2)
====================================================

fig_p15_0 (ISO 2372 / VDI 2056) tablosuna gore titresim hizi RMS (mm/s)
degerini makine sinifina (Class I-IV) bagli Zone A/B/C/D kararina cevirir.

NOT: Bu node HENUZ ana build_machine_expert_graph() akisina BAGLI DEGIL.
Faz 6'da (Adim 6.4) 'iso_severity_zone == "D"' durumunda dogrudan acil
durum ajanina yonlendiren conditional edge eklendiginde, bu node
VibraDiagMainState uzerinden akisa dahil edilecek. Simdilik bagimsiz
cagirilabilir/test edilebilir bir node olarak hazir.

Girdi (state):
    machine_class : str|int          (orn. "Class II", "2", "II")
    vibration_velocity_rms_mm_s : float
    severity_table : dict, opsiyonel  (verilmezse JSON'dan yuklenir)
    table_chunks_path : str, opsiyonel

Cikti (state guncellemesi):
    iso_severity_zone : "A"|"B"|"C"|"D"
    iso_severity_meaning : str        (orn. "Just Tolerable")
    iso_severity_range_mm_s : (float, float)
    severity_table : dict             (sonraki cagrilar icin cache)
"""

from __future__ import annotations

from typing import TypedDict

from deterministic_tools.json_rule_engine import (
    DEFAULT_TABLE_CHUNKS_PATH,
    classify_severity,
    load_chunks,
    load_severity_table,
)


class IsoEvaluatorState(TypedDict, total=False):
    machine_class: str
    vibration_velocity_rms_mm_s: float
    severity_table: dict
    table_chunks_path: str
    iso_severity_zone: str
    iso_severity_meaning: str
    iso_severity_range_mm_s: tuple
    iso_severity_machine_description: str


def iso_evaluator_node(state: IsoEvaluatorState) -> dict:
    """fig_p15_0 tablosuna gore RMS hiz degerini Zone A/B/C/D'ye cevirir."""
    severity_table = state.get("severity_table")
    if severity_table is None:
        chunks_path = state.get("table_chunks_path", DEFAULT_TABLE_CHUNKS_PATH)
        chunks = load_chunks(chunks_path)
        severity_table = load_severity_table(chunks)

    result = classify_severity(
        machine_class=state["machine_class"],
        rms_mm_s=state["vibration_velocity_rms_mm_s"],
        severity_table=severity_table,
    )

    return {
        "iso_severity_zone": result["zone"],
        "iso_severity_meaning": result["meaning"],
        "iso_severity_range_mm_s": result["range_mm_s"],
        "iso_severity_machine_description": result["machine_description"],
        "severity_table": severity_table,  
    }
