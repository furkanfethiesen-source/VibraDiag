


from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph
from scipy.signal.windows import hann

from deterministic_tools.empirical_band_rules import select_band_empirical
from deterministic_tools.fault_localization import analyze_multichannel, localize_fault
from deterministic_tools.json_rule_engine import (
    DEFAULT_VISUAL_CHUNKS_PATH,
    cross_check_ratio_pattern,
    load_chunks,
    load_spectral_patterns,
)
from deterministic_tools.reader import LoadedSignal
from deterministic_tools.reciprocating_faults import diagnose_reciprocating
from deterministic_tools.signal_processing import (
    is_kurtogram_reliable,
    match_ratio_pattern,
    select_band,
)


from schemas.states import SignalProcessingInternalState


def machine_type_router(state: SignalProcessingInternalState) -> str:
    """Conditional entry point: state['machine_type']'a gore dallanir.
    Gecersiz/eksik machine_type acik bir hata firlatir — sessizce
    varsayilan bir dala dusmek yanlis teshise yol acabilir."""
    machine_type = state.get("machine_type")
    if machine_type not in ("rotating", "reciprocating"):
        raise ValueError(
            f"machine_type_router: gecersiz veya eksik machine_type={machine_type!r}. "
            "'rotating' veya 'reciprocating' olmali."
        )
    return machine_type


def kurtogram_node(state: SignalProcessingInternalState) -> dict:
    """Birincil kanalda (DE varsa DE, yoksa ilk kanal) kurtogram calistirir
    ve guvenilirligini degerlendirir. Rotating dalinin HER ZAMAN ilk adimi;
    fallback'e girip girmeyecegine buradaki sonuc karar verir."""
    loaded: LoadedSignal = state["loaded_signal"]
    primary_channel = "DE" if "DE" in loaded.channels else next(iter(loaded.channels))
    signal = loaded.channels[primary_channel]

    band, kurtogram = select_band(signal, loaded.fs)
    reliable = is_kurtogram_reliable(signal, loaded.fs, kurtogram)

    return {
        "primary_channel": primary_channel,
        "kurtogram_band": band,
        "kurtogram": kurtogram,
        "kurtogram_reliable": reliable,
    }


def kurtogram_reliability_router(state: SignalProcessingInternalState) -> str:
    return "rotating_expert" if state["kurtogram_reliable"] else "fallback_band"


def fallback_band_node(state: SignalProcessingInternalState) -> dict:
    """Kurtogram guvenilmez bulundugunda, RPM'e gore table_p29_vision'dan
    ampirik bir zarf-filtresi bandi secer. Bu bant kurtogram bandinin
    YERINE gecer (rotating_expert_node override_band olarak kullanir)."""
    result = select_band_empirical(state["rpm"])
    return {
        "final_band": result["band_hz"],
        "band_source": result["source"],
        "band_filter_used": result["filter"],
    }


def rotating_expert_node(state: SignalProcessingInternalState) -> dict:
    """Donel makineler icin iki analiz dalini birlestirir:
    1) Zarf spektrumu (rulman arizalari: BPFO/BPFI/BSF/FTF) - coklu kanal,
    kurtogram VEYA fallback bandiyla (state['final_band'] varsa o,
    yoksa kurtogram_band).
    2) Dogrudan spektrum (unbalance/misalignment/looseness) - birincil
    kanal uzerinden match_ratio_pattern.
    """
    loaded: LoadedSignal = state["loaded_signal"]
    band = state.get("final_band") or state["kurtogram_band"]

    channel_analyses = analyze_multichannel(
        loaded, rpm=state["rpm"], n_balls=state["n_balls"],
        ball_diameter=state["ball_diameter"], pitch_diameter=state["pitch_diameter"],
        contact_angle_deg=state.get("contact_angle_deg", 0.0),
        override_band=band,
    )
    envelope_diagnosis = {ch: a.fault_results for ch, a in channel_analyses.items()}
    localization = localize_fault(channel_analyses)

    primary_signal = loaded.channels[state["primary_channel"]]
    direct_freqs = np.fft.rfftfreq(len(primary_signal), 1 / loaded.fs)
    direct_mag = np.abs(np.fft.rfft(primary_signal * hann(len(primary_signal))))
    fr = state["rpm"] / 60.0
    direct_spectrum_diagnosis = match_ratio_pattern(direct_freqs, direct_mag, fr)

    try:
        visual_chunks = load_chunks(DEFAULT_VISUAL_CHUNKS_PATH)
        spectral_patterns = load_spectral_patterns(visual_chunks)
        direct_spectrum_diagnosis["visual_pattern_cross_check"] = cross_check_ratio_pattern(
            direct_spectrum_diagnosis, spectral_patterns,
        )
    except FileNotFoundError:
        pass

    return {
        "envelope_diagnosis": envelope_diagnosis,
        "fault_localization": localization,
        "direct_spectrum_diagnosis": direct_spectrum_diagnosis,
    }


def reciprocating_expert_node(state: SignalProcessingInternalState) -> dict:
    """Pistonlu makineler icin table_p47_0+p48 denklemlerini hesaplar ve
    birincil kanalin dogrudan spektrumuyla eslestirir."""
    loaded: LoadedSignal = state["loaded_signal"]
    primary_channel = "DE" if "DE" in loaded.channels else next(iter(loaded.channels))
    signal = loaded.channels[primary_channel]

    direct_freqs = np.fft.rfftfreq(len(signal), 1 / loaded.fs)
    direct_mag = np.abs(np.fft.rfft(signal * hann(len(signal))))

    diagnosis = diagnose_reciprocating(
        direct_freqs, direct_mag, rpm=state["rpm"],
        n_cylinders=state["n_cylinders"],
        stroke_type=state.get("stroke_type", "4-stroke"),
    )

    return {"primary_channel": primary_channel, "reciprocating_diagnosis": diagnosis}




