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

import numpy as np
from scipy.integrate import simpson
from scipy.signal import butter, sosfiltfilt, hilbert, stft, find_peaks, detrend
from scipy.signal.windows import hann
from scipy.stats import kurtosis


def select_band(signal: np.ndarray, fs: float, nlevels: int = 5, nperseg: int = 1024):
    """
    Spektral kurtosis (kurtogram) kullanarak en darbeli (impulsive) bilginin
    bulundugu frekans bandini otomatik olarak belirler.

    Adim 3.5: bant-ici kurtosis hesabi artik her seviyede TEK bir
    np.add.reduceat + vektorize kurtosis cagrisi ile yapiliyor (eskiden
    her bant icin ayri Python dongusu vardi).

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

    best_kurt = -np.inf
    best_band = (freqs[0], freqs[-1])
    kurtogram = []

    for level in range(1, nlevels + 1):
        n_bands = 2 ** level
        edges = np.linspace(0, n_freq_bins, n_bands + 1, dtype=int)
        band_sizes = np.diff(edges)

        band_sums = np.add.reduceat(magnitude, edges[:-1], axis=0)
        level_kurts = kurtosis(band_sums, axis=1, fisher=True)
        level_kurts = np.where(band_sizes >= 2, level_kurts, -np.inf)

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
                    order: int = 4, guard_ratio: float = 0.1):
    """
    select_band()'in buldugu bandi kullanarak sinyale sifir-fazli
    (zero-phase) IIR (Butterworth) bant geciren filtre uygular.
    """
    nyquist = fs / 2
    low, high = band

    if high >= nyquist:
        high = nyquist * (1 - guard_ratio)
    if low <= 0:
        low = 1.0
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
                remove_dc: bool = True):
    """
    hilbert_envelope() ciktisinin kendi spektrumunu (envelope spectrum) cikarir.
    Pencereleme (Hann) spektral sizintiyi azaltir.

    Adim 3.4: zero_pad_factor varsayilani 4'ten 1'e indirildi. Bin-alti
    frekans hassasiyeti artik zero-padding yerine match_peaks/
    match_ratio_pattern icindeki 3-nokta parabolik enterpolasyonla
    saglaniyor — ayni hassasiyet, daha kucuk FFT boyutu (daha az bellek/
    hesap). Ihtiyac halinde zero_pad_factor eskisi gibi >1 verilebilir.

    Adim 3.5: gereksiz .copy() kaldirildi — `x * window` zaten yeni bir
    array urettigi icin orijinal `envelope.copy()` hicbir zaman islevsel
    olarak gerekli degildi.
    """
    n = len(envelope)
    x = envelope - np.mean(envelope) if remove_dc else envelope

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

    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


def _dynamic_tolerance(harmonic, base_tolerance_hz: float, alpha: float):
    """Tolerance(h) = base * (1 + alpha*(h-1)).

    Skaler ya da numpy array `harmonic` ile calisir (dogal olarak
    vektorize — Adim 3.5). Ust harmoniklerde STFT bin kaymasi ve hafif
    RPM dalgalanmasi biriktigi icin sabit tolerans yerine derece ile
    buyuyen tolerans daha gercekci eslesme saglar.
    """
    return base_tolerance_hz * (1 + alpha * (np.asarray(harmonic, dtype=float) - 1))


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
                tolerance_alpha: float = 0.15, prominence_ratio: float = 0.05):
    """
    envelope_fft() spektrumundaki tepe noktalarini calc_fault_freqs()'ten
    gelen karakteristik frekanslarla (ve harmonikleriyle) eslestirir, her
    ariza tipi icin bir siddet skoru uretir.

    Bir harmonigin "eslesmis" sayilmasi icin (confidence) hala gercek bir
    spektral tepe noktasina yakin olmasi gerekiyor (find_peaks); ancak
    raporlanan genlik/siddet artik o tepenin tek bin'lik degeri degil,
    dinamik-toleranslu bant icindeki RMS enerjisi (Adim 3.3) ve tepe
    frekansi parabolik enterpolasyonla rafine edilmis halidir (Adim 3.4).
    Harmonik dongusu numpy broadcasting ile vektorize edildi (Adim 3.5).
    """
    min_prominence = magnitude.max() * prominence_ratio
    peak_idx, _ = find_peaks(magnitude, prominence=min_prominence)
    peak_freqs = freqs[peak_idx]

    if len(peak_freqs) == 0:
        return {name: {"severity": 0.0, "matched_harmonics": [], "confidence": 0.0}
                for name in fault_freqs}

    baseline = np.median(magnitude[magnitude > 0])
    results = {}

    for fault_name, base_freq in fault_freqs.items():
        harmonics = np.arange(1, n_harmonics + 1)
        target_freqs = base_freq * harmonics
        valid = target_freqs < freqs[-1]
        harmonics, target_freqs = harmonics[valid], target_freqs[valid]

        matched_harmonics = []
        total_energy = 0.0

        if len(target_freqs) > 0:
            diff_matrix = np.abs(peak_freqs[None, :] - target_freqs[:, None])
            closest_peak_local_idx = np.argmin(diff_matrix, axis=1)
            closest_diffs = diff_matrix[np.arange(len(target_freqs)), closest_peak_local_idx]
            tolerances = _dynamic_tolerance(harmonics, tolerance_hz, tolerance_alpha)

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

        severity = total_energy / baseline if baseline > 0 else 0.0
        confidence = len(matched_harmonics) / n_harmonics

        results[fault_name] = {
            "severity": round(severity, 2),
            "matched_harmonics": matched_harmonics,
            "confidence": round(confidence, 2),
        }

    return results


def match_named_frequencies(freqs: np.ndarray, magnitude: np.ndarray,
                            named_freqs: dict, tolerance_hz: float = 2.0,
                            prominence_ratio: float = 0.05) -> dict:
    """match_peaks'in genellemesi: named_freqs'teki her frekans TEK BASINA
    mutlak bir hedeftir (bir taban frekansin harmonik serisi degil).

    Kullanim alani: pistonlu makine ariza frekanslari (table_p47_0+p48)
    gibi, her biri farkli bir fiziksel mekanizmaya (piston slap, misfiring
    vb.) karsilik gelen, birbiriyle harmonik iliskisi olmayan isimli
    frekans setleri.

    "Eslesme" (matched=True) icin gercek bir spektral tepe (find_peaks)
    gerekiyor; raporlanan genlik/siddet ise match_peaks'teki gibi
    dinamik-toleranslu bant-RMS enerjisi (Faz 3, Adim 3.3).
    """
    min_prominence = magnitude.max() * prominence_ratio
    peak_idx, _ = find_peaks(magnitude, prominence=min_prominence)
    peak_freqs = freqs[peak_idx]
    baseline = np.median(magnitude[magnitude > 0])

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

        band_amp = _band_rms_energy(freqs, magnitude, target_freq, tolerance_hz)
        severity = band_amp / baseline if baseline > 0 else 0.0

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
    misalignment / looseness icin oran tabanli desen eslestirmesi yapar.

    Adim 3.2/3.3: harmonik genlikleri artik dinamik toleranslu bant-RMS
    enerjisiyle hesaplaniyor (find_peaks + en-yakin-tepe arama yerine).
    NOT (semantik degisiklik): eski yontemde bir harmonikte gercek bir
    spektral tepe yoksa genlik tam 0 donuyordu; RMS-bant yontemi o
    bolgedeki gurultu tabanini da yakaladigi icin kucuk-ama-sifir-olmayan
    degerler dondurebilir. Bu fiziksel olarak daha dogru (enerji tabanli)
    ama eski severity sayilarinla birebir olceklenmez.

    NOT (bilinen limitasyon, degismedi): misalignment ve looseness spektral
    imzalari birbirine benzeyebilir (ikisi de ust harmonikleri yukseltir);
    tek kanalli bir olcumle bu ikisi her zaman guvenilir sekilde
    ayristirilamaz. fault_localization.check_coupling_phase() ile faz
    bilgisi eklenmesi onerilir (bkz. Faz 2).
    """
    baseline = np.median(magnitude[magnitude > 0])

    harmonics = np.arange(1, n_harmonics + 1)
    target_freqs = fr * harmonics
    valid = target_freqs < freqs[-1]
    harmonics, target_freqs = harmonics[valid], target_freqs[valid]
    tolerances = _dynamic_tolerance(harmonics, tolerance_hz, tolerance_alpha)

    amps = {
        int(h): _band_rms_energy(freqs, magnitude, tf, tol)
        for h, tf, tol in zip(harmonics, target_freqs, tolerances)
    }

    subharmonic_tol = float(_dynamic_tolerance(1, tolerance_hz, tolerance_alpha))
    subharmonic_amp = _band_rms_energy(freqs, magnitude, fr * 0.5, subharmonic_tol)

    a1 = amps.get(1, 0.0)

    unbalance_ratio_penalty = (
        sum(amps.get(h, 0.0) for h in range(2, n_harmonics + 1)) / a1
        if a1 > 0 else np.inf
    )
    unbalance_severity = (a1 / baseline) if baseline > 0 else 0.0
    unbalance_confidence = max(0.0, 1.0 - unbalance_ratio_penalty) if a1 > 0 else 0.0

    a2 = amps.get(2, 0.0)
    misalignment_ratio = (a2 / a1) if a1 > 0 else 0.0
    misalignment_severity = (a2 / baseline) if baseline > 0 else 0.0
    misalignment_confidence = min(1.0, misalignment_ratio / 0.5) if misalignment_ratio > 0 else 0.0

    strong_harmonics = [h for h, a in amps.items()
                        if a > 0 and (a / baseline if baseline > 0 else 0) > 3]
    looseness_energy = sum(amps.values()) + subharmonic_amp
    looseness_severity = (looseness_energy / baseline) if baseline > 0 else 0.0
    looseness_confidence = min(1.0, len(strong_harmonics) / n_harmonics)
    if subharmonic_amp > 0:
        looseness_confidence = min(1.0, looseness_confidence + 0.2)

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

