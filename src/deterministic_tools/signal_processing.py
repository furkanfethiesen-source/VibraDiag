"""
VibraDiag - Zarf Analizi ve Dogrudan Spektrum Analizi Fonksiyonlari
=====================================================================

Bu modul, rulman ve mekanik ariza teshisi icin gelistirilen VibraDiag
projesinin sinyal isleme katmanini olusturan fonksiyonlari bir arada toplar.

Sinyal yukleme artik bu modulun sorumlulugunda DEGIL — readers.SignalReaderFactory
kullanilir (.wav/.mat/.csv, coklu kanal destegi). Bu modul, zaten yuklenmis
bir np.ndarray + fs uzerinden calisan saf sinyal isleme fonksiyonlarini icerir.

Iki analiz dali icerir:
1) Zarf analizi hatti (rulman arizalari icin):
    readers.SignalReaderFactory.load -> select_band -> bandpass_filter
    -> hilbert_envelope -> envelope_fft -> calc_fault_freqs -> match_peaks

2) Dogrudan spektrum analizi hatti (unbalance/misalignment/looseness icin):
    readers.SignalReaderFactory.load -> (dogrudan FFT) -> match_ratio_pattern

Bagimliliklar: numpy, scipy
"""

from typing import Any

import numpy as np
from scipy.integrate import simpson
from scipy.signal import butter, sosfiltfilt, hilbert, stft, find_peaks, detrend
from scipy.signal.windows import hann
from scipy.stats import kurtosis


def select_band(signal: np.ndarray, fs: float, nlevels: int = 5, nperseg: int = 1024,
                nyquist_guard_ratio: float = 0.15):
    """
    Spektral kurtosis (kurtogram) kullanarak en darbeli (impulsive) bilginin
    bulundugu frekans bandini otomatik olarak belirler.

    Kenar artefaktlarini (edge effects) ve Nyquist gecis bolgesindeki sahte yuksek
    kurtosis tepe noktalarini onlemek icin ust frekans siniri (1 - nyquist_guard_ratio) * Nyquist
    ile sinirlandirilir.

    Returns
    -------
    band : tuple(float, float)
        Secilen bandin (alt_frekans, ust_frekans) sinirlari (Hz)
    kurtogram : list
        Gorsellestirme/analiz icin tum (seviye, bant) kurtosis degerleri
    """
    freqs, _, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    magnitude = np.abs(Zxx)
    n_freq_bins = magnitude.shape[0]
    nyquist = fs / 2.0
    max_freq_limit = (1.0 - nyquist_guard_ratio) * nyquist

    best_kurt = -np.inf
    best_band = (freqs[0], freqs[-1])
    kurtogram = []

    for level in range(1, nlevels + 1):
        n_bands = 2 ** level
        edges = np.linspace(0, n_freq_bins, n_bands + 1, dtype=int)
        band_sizes = np.diff(edges)

        band_sums = np.add.reduceat(magnitude, edges[:-1], axis=0)
        level_kurts = kurtosis(band_sums, axis=1, fisher=True)

        for b_idx in range(n_bands):
            lo, hi = edges[b_idx], edges[b_idx + 1]
            band_high = freqs[hi - 1] if hi > 0 else 0.0
            if band_high > max_freq_limit or band_sizes[b_idx] < 2:
                level_kurts[b_idx] = -np.inf

        level_best_idx = int(np.argmax(level_kurts))
        if level_kurts[level_best_idx] > best_kurt:
            best_kurt = float(level_kurts[level_best_idx])
            lo, hi = edges[level_best_idx], edges[level_best_idx + 1]
            best_band = (freqs[lo], freqs[hi - 1])

        kurtogram.append(level_kurts.tolist())

    return best_band, kurtogram


def is_kurtogram_reliable(signal: np.ndarray, fs: float, kurtogram: list,
                        nlevels: int = 5, nperseg: int = 1024,
                        n_surrogates: int = 5, z_threshold: float = 3.0,
                        rng: np.random.RandomState | None = None) -> bool:
    """select_band()'in urettigi kurtogramin GERCEK bir darbeli
    (impulsive) yapi mi yakaladigini, yoksa sadece istatistiksel sans
    eseri mi yuksek kurtosis urettigini surrogate-data (karistirma-null)
    testiyle ayirt eder (Faz 4 / Fallback Node tetikleyicisi).

    NEDEN basit bir sabit-esik YETERLI DEGIL (ilk denemede oyle yazilmisti,
    test sirasinda yanlis ciktigi icin duzeltildi): band_envelope (STFT
    genlik toplami) dogasi geregi Gaussian degildir (Rayleigh-benzeri
    dagilim), bu yuzden SAF GURULTUDE BILE kurtosis degerleri 0'a yakin
    cikmiyor — ozellikle kisa sinyal / az STFT zaman cercevesi durumunda
    iki haneli degerlere ulasabiliyor. 'kurtosis > sabit_esik' testi bu
    yuzden gurultude de sikca True donuyordu (yanlis pozitif); olcumle
    dogrulandi.

    Yontem: sinyalin zaman-domain orneklerini rastgele karistirarak
    (np.random.permutation) n_surrogates adet "surrogate" sinyal uretilir.
    Karistirma olasi periyodik/darbeli zamansal yapiyi yok eder ama genlik
    dagilimini korur — yani surrogate'lerin kurtogram tepe degeri, "hicbir
    gercek darbeli yapi olmasaydi sans eseri ne kadar yuksek kurtosis
    cikardi" sorusuna cevap veren bir null-dagilim olusturur. Gercek
    sinyalin tepe kurtosis'i bu null dagilimdan z_threshold kadar standart
    sapma yukarida ise kurtogram guvenilir kabul edilir.

    NOT: n_surrogates kadar ekstra select_band() cagrisi yapar (maliyetli
    olabilir); buyuk sinyallerde n_surrogates dusurulebilir.
    """
    def _peak_kurtosis(kg: list) -> float:
        flat = np.concatenate([np.asarray(level, dtype=float) for level in kg])
        finite = flat[np.isfinite(flat)]
        return float(finite.max()) if finite.size > 0 else -np.inf

    real_peak = _peak_kurtosis(kurtogram)
    if not np.isfinite(real_peak):
        return False

    rng = rng if rng is not None else np.random.RandomState()
    surrogate_peaks = np.array([
        _peak_kurtosis(select_band(rng.permutation(signal), fs,
                                    nlevels=nlevels, nperseg=nperseg)[1])
        for _ in range(n_surrogates)
    ])

    surrogate_std = float(surrogate_peaks.std())
    if surrogate_std < 1e-9:
        return real_peak > float(surrogate_peaks.mean())

    z = (real_peak - float(surrogate_peaks.mean())) / surrogate_std
    return z >= z_threshold


def bandpass_filter(signal: np.ndarray, fs: float, band: tuple,
                    order: int = 4, guard_ratio: float = 0.02):
    """
    select_band()'in buldugu bandi kullanarak sinyale sifir-fazli
    (zero-phase) IIR (Butterworth) bant geciren filtre uygular.
    """
    nyquist = fs / 2
    low, high = float(band[0]), float(band[1])

    max_safe_high = nyquist * (1.0 - guard_ratio)
    if high >= max_safe_high:
        high = max_safe_high

    if low <= 0:
        low = 1.0

    if low >= high:
        low = max(1.0, high - max(50.0, (band[1] - band[0]) * 0.5))
        if low >= high:
            low = max(1.0, high * 0.8)

    if low >= high:
        raise ValueError(f"Gecersiz bant: alt sinir ({low} Hz) >= ust sinir ({high} Hz)")

    low_norm = low / nyquist
    high_norm = high / nyquist

    sos = butter(order, [low_norm, high_norm], btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, signal)
    return filtered


def hilbert_envelope(filtered_signal: np.ndarray) -> np.ndarray:
    """
    bandpass_filter() ciktisina Hilbert donusumu uygulayarak sinyalin
    zarfini (envelope / anlik genlik) cikarir.
    """
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    return envelope


def envelope_fft(envelope: np.ndarray, fs: float, zero_pad_factor: int = 1,
                remove_dc: bool = True, high_pass_cutoff_hz: float | None = 2.0):
    """
    hilbert_envelope() ciktisinin kendi spektrumunu (envelope spectrum) cikarir.
    DC ofset ve asiri dusuk frekansli zemin kaymalarini (drift) gidermek icin
    yuksek geciren filtreleme ve Hann pencerelemesi uygular.
    """
    n = len(envelope)
    x = detrend(envelope, type='linear') if remove_dc else envelope.copy()
    if remove_dc:
        x = x - np.mean(x)

    if high_pass_cutoff_hz is not None and high_pass_cutoff_hz > 0 and (high_pass_cutoff_hz < fs / 2.0):
        try:
            sos = butter(2, high_pass_cutoff_hz / (fs / 2.0), btype='highpass', output='sos')
            x = sosfiltfilt(sos, x)
        except Exception:
            pass

    window = hann(n)
    x_windowed = x * window
    amplitude_correction = 1.0 / np.mean(window)
    x_windowed *= amplitude_correction

    n_fft = n * zero_pad_factor
    spectrum = np.fft.rfft(x_windowed, n=n_fft)
    magnitude = np.abs(spectrum) / n
    magnitude[1:-1] *= 2

    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    return freqs, magnitude


def calc_fault_freqs(rpm: float, n_balls: int, ball_diameter: float,
                    pitch_diameter: float, contact_angle_deg: float = 0.0):
    """
    Rulman geometrisi ve sart RPM'inden karakteristik ariza frekanslarini
    hesaplar: BPFO (dis bilezik), BPFI (ic bilezik), BSF (bilye), FTF (kafes).
    """
    if n_balls < 2:
        raise ValueError("n_balls en az 2 olmali")
    if ball_diameter <= 0 or pitch_diameter <= 0:
        raise ValueError("ball_diameter ve pitch_diameter pozitif olmali")
    if ball_diameter >= pitch_diameter:
        raise ValueError("ball_diameter, pitch_diameter'dan kucuk olmali")

    fr = rpm / 60.0
    ratio = ball_diameter / pitch_diameter
    phi = np.deg2rad(contact_angle_deg)
    cos_phi = np.cos(phi)

    bpfo = (n_balls / 2.0) * fr * (1 - ratio * cos_phi)
    bpfi = (n_balls / 2.0) * fr * (1 + ratio * cos_phi)
    bsf = (pitch_diameter / (2.0 * ball_diameter)) * fr * (1 - (ratio * cos_phi) ** 2)
    ftf = (fr / 2.0) * (1 - ratio * cos_phi)

    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf, "2BSF": 2.0 * bsf}


def _dynamic_tolerance(
    harmonic,
    base_freq: float = 1.0,
    min_abs_tol_hz: float = 0.5,
    tolerance_ratio: float = 0.015,
    alpha: float = 0.10,
    tolerance_hz: float | None = None,
):
    """
    Tolerance(h) = max(min_abs_tol_hz, (base_freq * h) * tolerance_ratio) * (1 + alpha * (h - 1)).

    Works with scalar or numpy array `harmonic`.
    Proportional tolerance prevents over-wide windows for low-frequency faults (FTF ~12 Hz)
    while maintaining appropriate margins for high-frequency faults (BPFI ~162 Hz).
    """
    harm = np.asarray(harmonic, dtype=float)
    if tolerance_hz is not None and base_freq <= 1.0:
        base_tol = tolerance_hz
    else:
        freq_targets = base_freq * harm
        base_tol = np.maximum(min_abs_tol_hz, freq_targets * tolerance_ratio)

    return base_tol * (1.0 + alpha * (harm - 1.0))


def _get_local_baseline(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    target_freq: float,
    fr: float = 29.95,
) -> float:
    """
    Calculates a localized noise floor baseline around target_freq.
    For low-frequency regions (0 - 2.5 * fr), uses the 0.5 Hz - 3.0 * fr window to avoid
    exaggerated severity scores caused by DC offset and 1X energy leakage.
    For higher frequencies, uses a localized window [f - 5*fr, f + 5*fr].
    """
    if len(magnitude) == 0:
        return 1.0

    if target_freq <= 2.5 * fr:
        mask = (freqs >= 0.5) & (freqs <= max(3.0 * fr, 100.0))
    else:
        half_win = max(5.0 * fr, 50.0)
        mask = (freqs >= max(0.5, target_freq - half_win)) & (freqs <= target_freq + half_win)

    sub_mag = magnitude[mask]
    if len(sub_mag) > 0:
        pos = sub_mag[sub_mag > 0]
        if len(pos) > 0:
            med = float(np.median(pos))
            if med > 1e-9:
                return med

    global_pos = magnitude[magnitude > 0]
    return float(np.median(global_pos)) if len(global_pos) > 0 else 1.0


def _check_sidebands(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    peak_freqs: np.ndarray,
    center_freq: float,
    fr: float,
    fault_name: str = "BPFI",
    ftf: float | None = None,
    min_abs_tol_hz: float = 0.5,
    tolerance_ratio: float = 0.015,
) -> dict[str, Any]:
    """
    Checks for modulation sidebands:
    - BPFI: checks shaft speed modulation (center +- fr, center +- 2*fr)
    - BSF: checks cage speed modulation (center +- FTF, center +- 2*FTF, and center +- fr)
    Returns sideband matching status, matched frequencies, and total sideband energy.
    """
    sideband_orders = [-2, -1, 1, 2]
    matched_sidebands = []
    total_sideband_energy = 0.0

    # Determine modulation frequencies to test
    mod_configs: list[tuple[float, str]] = []
    if fault_name == "BSF":
        cage_mod = ftf if (ftf is not None and ftf > 0) else (fr * 0.4)
        mod_configs.append((cage_mod, "FTF"))
        mod_configs.append((fr, "1X"))
    else:
        mod_configs.append((fr, "1X"))

    # For BSF, check sidebands around both 2xBSF and 1xBSF
    center_freqs = [center_freq, center_freq * 2.0] if fault_name == "BSF" else [center_freq]

    for c_freq in center_freqs:
        for mod_spacing, mod_label in mod_configs:
            for order in sideband_orders:
                sb_target = c_freq + (order * mod_spacing)
                if sb_target < 0.5 or sb_target >= freqs[-1]:
                    continue

                sb_tol = max(min_abs_tol_hz, sb_target * tolerance_ratio)
                diffs = np.abs(peak_freqs - sb_target) if len(peak_freqs) > 0 else np.array([])
                if len(diffs) > 0:
                    min_diff_idx = int(np.argmin(diffs))
                    if diffs[min_diff_idx] <= sb_tol:
                        matched_sb_freq = float(peak_freqs[min_diff_idx])
                        sb_amp = _band_rms_energy(freqs, magnitude, sb_target, sb_tol)
                        matched_sidebands.append({
                            "order": f"{order:+d}{mod_label} ({c_freq:.1f}Hz)",
                            "target_hz": round(sb_target, 2),
                            "found_hz": round(matched_sb_freq, 2),
                            "amplitude": round(sb_amp, 4),
                        })
                        total_sideband_energy += sb_amp

    has_sidebands = len(matched_sidebands) >= 1
    return {
        "sidebands_detected": has_sidebands,
        "n_sidebands_matched": len(matched_sidebands),
        "matched_sidebands": matched_sidebands,
        "total_sideband_energy": round(total_sideband_energy, 4),
    }


def _parabolic_peak_refine(magnitude: np.ndarray, idx: int, freqs: np.ndarray):
    """3-nokta parabolik enterpolasyon: bin cozunurlugunun altinda tepe
    frekansi/genligi tahmini (zero-padding'e alternatif, Adim 3.4)."""
    if idx <= 0 or idx >= len(magnitude) - 1:
        return float(freqs[idx]), float(magnitude[idx])

    y_left, y_center, y_right = magnitude[idx - 1], magnitude[idx], magnitude[idx + 1]
    denom = y_left - 2 * y_center + y_right
    if denom == 0:
        return float(freqs[idx]), float(magnitude[idx])

    delta = 0.5 * (y_left - y_right) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    bin_width = freqs[idx + 1] - freqs[idx]
    refined_freq = float(freqs[idx] + delta * bin_width)
    refined_amp = float(y_center - 0.25 * (y_left - y_right) * delta)
    return refined_freq, refined_amp


def _band_rms_energy(freqs: np.ndarray, magnitude: np.ndarray,
                    center_freq: float, half_width_hz: float) -> float:
    """Hedef frekans etrafindaki [center-hw, center+hw] bandinda
    simpson-entegre RMS enerjisi (Adim 3.3). Tek bir FFT bin'inin tepe
    genligi yerine bandin toplam enerjisini kullanir — hafif frekans
    kaymalarina ve spektral sizintiya karsi daha gurbuz (robust)."""
    lo, hi = center_freq - half_width_hz, center_freq + half_width_hz
    mask = (freqs >= lo) & (freqs <= hi)

    if mask.sum() < 2:
        idx = int(np.argmin(np.abs(freqs - center_freq)))
        return float(magnitude[idx])

    band_freqs = freqs[mask]
    band_power = magnitude[mask] ** 2
    energy = simpson(band_power, x=band_freqs)
    width = band_freqs[-1] - band_freqs[0]
    return float(np.sqrt(energy / width)) if width > 0 else float(magnitude[mask].max())


def match_peaks(freqs: np.ndarray, magnitude: np.ndarray, fault_freqs: dict,
                n_harmonics: int = 3, tolerance_hz: float = 2.0,
                tolerance_alpha: float = 0.10, prominence_ratio: float = 0.05,
                fr: float = 29.95, min_abs_tol_hz: float = 0.5,
                tolerance_ratio: float = 0.015):
    """
    2 Kademeli Hibrit (Gated Band-RMS) Zarf Spektrumu Tepe ve Harmonik Eşleştiricisi.
    Kademe 1 (Gating): Hedef frekans tolerans penceresinde gürültü zeminini aşan gerçek bir tepe var mı kontrol eder.
    Kademe 2 (Bant-RMS): Tepe varlığı doğrulandığında, arıza şiddetini Simpson entegrasyonlu bant enerjisiyle hesaplar.
    """
    baseline = np.median(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1.0
    min_prominence = max(magnitude.max() * prominence_ratio, baseline * 1.5)
    peak_idx, _ = find_peaks(magnitude, prominence=min_prominence)
    peak_freqs = freqs[peak_idx] if len(peak_idx) > 0 else np.array([])

    ftf_freq = fault_freqs.get("FTF")

    if len(peak_freqs) == 0:
        return {name: {
            "severity": 0.0, "matched_harmonics": [], "confidence": 0.0,
            "sidebands": {"sidebands_detected": False, "n_sidebands_matched": 0, "matched_sidebands": []}
        } for name in fault_freqs if name not in ("fr", "2BSF")}

    results = {}

    for fault_name, base_freq in fault_freqs.items():
        if fault_name in ("fr", "2BSF") or base_freq is None:
            continue

        harmonics = np.arange(1, n_harmonics + 1)
        target_freqs = base_freq * harmonics
        valid = target_freqs < freqs[-1]
        harmonics, target_freqs = harmonics[valid], target_freqs[valid]

        matched_harmonics = []
        total_energy = 0.0

        local_baseline = _get_local_baseline(freqs, magnitude, base_freq, fr=fr)

        if len(target_freqs) > 0 and len(peak_freqs) > 0:
            diff_matrix = np.abs(peak_freqs[None, :] - target_freqs[:, None])
            closest_peak_local_idx = np.argmin(diff_matrix, axis=1)
            closest_diffs = diff_matrix[np.arange(len(target_freqs)), closest_peak_local_idx]

            tolerances = _dynamic_tolerance(
                harmonics,
                base_freq=base_freq,
                min_abs_tol_hz=min_abs_tol_hz,
                tolerance_ratio=tolerance_ratio,
                alpha=tolerance_alpha,
            )

            for h, target_freq, local_idx, diff, tol in zip(
                harmonics, target_freqs, closest_peak_local_idx, closest_diffs, tolerances
            ):
                if diff <= tol:
                    global_peak_idx = peak_idx[local_idx]
                    refined_freq, _ = _parabolic_peak_refine(magnitude, global_peak_idx, freqs)
                    band_amp = _band_rms_energy(freqs, magnitude, target_freq, tol)

                    matched_harmonics.append({
                        "harmonic": int(h),
                        "target_hz": round(float(target_freq), 2),
                        "found_hz": round(refined_freq, 2),
                        "tolerance_hz": round(float(tol), 2),
                        "amplitude": round(band_amp, 4),
                    })
                    total_energy += band_amp

        # Check sidebands
        sideband_info = {"sidebands_detected": False, "n_sidebands_matched": 0, "matched_sidebands": []}
        if fault_name in ("BPFI", "BSF") and len(matched_harmonics) > 0:
            sideband_info = _check_sidebands(
                freqs=freqs,
                magnitude=magnitude,
                peak_freqs=peak_freqs,
                center_freq=base_freq,
                fr=fr,
                fault_name=fault_name,
                ftf=ftf_freq,
                min_abs_tol_hz=min_abs_tol_hz,
                tolerance_ratio=tolerance_ratio,
            )

        severity = total_energy / local_baseline if local_baseline > 0 else 0.0
        confidence = len(matched_harmonics) / n_harmonics

        results[fault_name] = {
            "severity": round(severity, 2),
            "matched_harmonics": matched_harmonics,
            "confidence": round(confidence, 2),
            "n_harmonics_matched": len(matched_harmonics),
            "sidebands": sideband_info,
        }

    return results


def match_named_frequencies(freqs: np.ndarray, magnitude: np.ndarray,
                            named_freqs: dict, tolerance_hz: float = 2.0,
                            prominence_ratio: float = 0.05) -> dict:
    """match_peaks'in genellemesi: named_freqs'teki her frekans TEK BASINA
    mutlak bir hedeftir (bir taban frekansin harmonik serisi degil).
    """
    baseline = np.median(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1.0
    min_prominence = max(magnitude.max() * prominence_ratio, baseline * 1.5)
    peak_idx, _ = find_peaks(magnitude, prominence=min_prominence)
    peak_freqs = freqs[peak_idx] if len(peak_idx) > 0 else np.array([])

    results = {}
    for name, target_freq in named_freqs.items():
        if target_freq is None or target_freq >= freqs[-1]:
            results[name] = {
                "matched": False, "target_hz": None,
                "found_hz": None, "amplitude": 0.0, "severity": 0.0,
            }
            continue

        matched = False
        found_hz = None
        if len(peak_freqs) > 0:
            diffs = np.abs(peak_freqs - target_freq)
            closest_local_idx = int(np.argmin(diffs))
            if diffs[closest_local_idx] <= tolerance_hz:
                matched = True
                global_idx = peak_idx[closest_local_idx]
                found_hz, _ = _parabolic_peak_refine(magnitude, global_idx, freqs)

        band_amp = _band_rms_energy(freqs, magnitude, target_freq, tolerance_hz) if matched else 0.0
        severity = band_amp / baseline if (baseline > 0 and matched) else 0.0

        results[name] = {
            "matched": matched,
            "target_hz": round(float(target_freq), 2),
            "found_hz": round(found_hz, 2) if found_hz is not None else None,
            "amplitude": round(band_amp, 4),
            "severity": round(severity, 2),
        }

    return results


def match_ratio_pattern(freqs: np.ndarray, magnitude: np.ndarray, fr: float,
                        n_harmonics: int = 5, tolerance_hz: float = 2.0,
                        tolerance_alpha: float = 0.15):
    """
    Ham sinyalin dogrudan spektrumundan (envelope_fft degil) unbalance /
    misalignment / looseness icin 2 Kademeli Hibrit (Gated Band-RMS) analizi yapar.

    Kademe 1: 1X, 2X, 3X... bilesenlerinde arka plan gurultusunu asan gercek bir tepe (peak) var mi kontrol eder.
    Kademe 2: Sadece tepe tespit edilen harmoniklerde Simpson Bant-RMS enerjisini hesaplar.
    """
    baseline = float(np.median(magnitude[magnitude > 0])) if np.any(magnitude > 0) else 1.0

    min_prom = baseline * 2.0
    peak_idx, _ = find_peaks(magnitude, prominence=min_prom)
    peak_freqs = freqs[peak_idx] if len(peak_idx) > 0 else np.array([])

    harmonics = np.arange(1, n_harmonics + 1)
    target_freqs = fr * harmonics
    valid = target_freqs < freqs[-1]
    harmonics, target_freqs = harmonics[valid], target_freqs[valid]
    tolerances = _dynamic_tolerance(harmonics, tolerance_hz, tolerance_alpha)

    amps = {}
    peaks_present = {}
    for h, tf, tol in zip(harmonics, target_freqs, tolerances):
        has_peak = False
        if len(peak_freqs) > 0:
            diffs = np.abs(peak_freqs - tf)
            if np.any(diffs <= tol):
                has_peak = True
        peaks_present[int(h)] = has_peak
        if has_peak:
            amps[int(h)] = _band_rms_energy(freqs, magnitude, tf, tol)
        else:
            amps[int(h)] = 0.0

    subharmonic_tol = float(_dynamic_tolerance(1, tolerance_hz, tolerance_alpha))
    has_sub = np.any(np.abs(peak_freqs - fr * 0.5) <= subharmonic_tol) if len(peak_freqs) > 0 else False
    subharmonic_amp = _band_rms_energy(freqs, magnitude, fr * 0.5, subharmonic_tol) if has_sub else 0.0

    a1 = amps.get(1, 0.0)
    a2 = amps.get(2, 0.0)

    if a1 > 0:
        higher_harm_energy = sum(amps.get(h, 0.0) for h in range(2, n_harmonics + 1))
        unbalance_ratio_penalty = higher_harm_energy / a1
        unbalance_severity = a1 / baseline if baseline > 0 else 0.0
        unbalance_confidence = max(0.0, 1.0 - unbalance_ratio_penalty)
    else:
        unbalance_severity = 0.0
        unbalance_confidence = 0.0

    if a2 > 0 and a1 > 0:
        misalignment_ratio = a2 / a1
        misalignment_severity = a2 / baseline if baseline > 0 else 0.0
        misalignment_confidence = min(1.0, misalignment_ratio / 0.5)
    elif a2 > 0:  
        misalignment_severity = a2 / baseline if baseline > 0 else 0.0
        misalignment_confidence = 0.50
    else:
        misalignment_severity = 0.0
        misalignment_confidence = 0.0

    strong_harmonics = [h for h, a in amps.items() if a > 0 and (a / baseline if baseline > 0 else 0) > 3.0]
    looseness_energy = sum(amps.values()) + subharmonic_amp
    if len(strong_harmonics) >= 2 or has_sub:
        looseness_severity = looseness_energy / baseline if baseline > 0 else 0.0
        looseness_confidence = min(1.0, len(strong_harmonics) / n_harmonics + (0.2 if has_sub else 0.0))
    else:
        looseness_severity = 0.0
        looseness_confidence = 0.0

    return {
        "unbalance": {"severity": round(unbalance_severity, 2),
                    "confidence": round(unbalance_confidence, 2)},
        "misalignment": {"severity": round(misalignment_severity, 2),
                        "confidence": round(misalignment_confidence, 2)},
        "looseness": {"severity": round(looseness_severity, 2),
                    "confidence": round(looseness_confidence, 2)},
        "harmonic_amplitudes": {h: round(a, 4) for h, a in amps.items()},
        "subharmonic_0.5x": round(subharmonic_amp, 4),
    }

