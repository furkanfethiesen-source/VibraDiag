
"""
VibraDiag - Polimorfik Coklu-Format Sinyal Okuyucu
=====================================================

Faz 1: .wav / .mat (v5 ve v7.3) / .csv formatlarini tek bir arayuz
(SignalReaderFactory.load) arkasinda birlestiren okuyucu katmani.

Tasarim notlari
----------------
- Her format icin ayri bir SignalReader alt sinifi var (OOP polymorphism).
- Donus tipi tum formatlarda ayni: LoadedSignal. CWRU .mat dosyalari
cok kanalli oldugu icin (DE/FE/BA) tekil sinyal donen .wav/.csv
okuyuculari da ayni sozlesmeye uyup channels={"main": ...} dondurur.
Bu, Faz 2'deki demux/fusion katmaninin tek bir veri sozlesmesiyle
calismasini saglar.
- .mat v7.3 dosyalari HDF5 formatindadir; scipy.io.loadmat bunlari
okuyamaz (NotImplementedError firlatir) -> h5py fallback.
- CWRU .mat dosyalarinda ornekleme hizi (fs) dosyanin icinde YOK.
Bu yuzden .mat okuyucusu icin expected_fs zorunludur.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, wavfile
from scipy.signal import detrend

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


@dataclass
class LoadedSignal:
    """Tum reader'larin dondugu ortak sonuc tipi."""
    channels: dict[str, np.ndarray]          # {"main": arr} veya {"DE": arr, "FE": arr, "BA": arr}
    fs: float
    rpm: float | None = None
    source_path: str = ""
    meta: dict = field(default_factory=dict)  # format-spesifik ek bilgi (orn. bulunan anahtar adlari)

    def __post_init__(self):
        for name, sig in self.channels.items():
            _validate_signal_array(sig, context=f"kanal '{name}'")



def _validate_fs(fs: float, expected_fs: float | None, min_fs: float) -> None:
    if fs < min_fs:
        raise ValueError(
            f"Ornekleme hizi cok dusuk: {fs} Hz (minimum {min_fs} Hz gerekli)."
        )
    if expected_fs is not None and not np.isclose(fs, expected_fs, rtol=0.01):
        raise ValueError(
            f"Beklenmeyen ornekleme hizi: dosyada {fs} Hz, beklenen {expected_fs} Hz"
        )


def _validate_signal_array(raw_signal: np.ndarray, context: str = "sinyal") -> None:
    if raw_signal.size == 0 or np.all(raw_signal == 0):
        raise ValueError(f"{context}: bos veya tamamen sifir — dosya bozuk olabilir")
    if np.any(np.isnan(raw_signal)) or np.any(np.isinf(raw_signal)):
        raise ValueError(f"{context}: NaN veya Inf degerler var")


def _clean(raw_signal: np.ndarray) -> np.ndarray:
    """DC bileseni kaldirir (mevcut load_signal ile birebir ayni islem)."""
    raw_signal = np.asarray(raw_signal, dtype=np.float64).squeeze()
    _validate_signal_array(raw_signal)
    return detrend(raw_signal, type="linear")



class SignalReader(ABC):
    """Tum format okuyucularinin uydugu ortak arayuz."""

    @abstractmethod
    def can_read(self, filepath: Path) -> bool:
        ...

    @abstractmethod
    def read(self, filepath: Path, expected_fs: float | None, min_fs: float) -> LoadedSignal:
        ...

class WavReader(SignalReader):
    def can_read(self, filepath: Path) -> bool:
        return filepath.suffix.lower() == ".wav"

    def read(self, filepath: Path, expected_fs: float | None, min_fs: float) -> LoadedSignal:
        fs, raw_signal = wavfile.read(filepath)
        _validate_fs(fs, expected_fs, min_fs)
        signal = _clean(raw_signal)
        return LoadedSignal(
            channels={"main": signal},
            fs=float(fs),
            source_path=str(filepath),
        )

_CHANNEL_PATTERN = re.compile(r"(DE|FE|BA)_time$")
_RPM_PATTERN = re.compile(r"RPM$", re.IGNORECASE)


class MatReader(SignalReader):
    def can_read(self, filepath: Path) -> bool:
        return filepath.suffix.lower() == ".mat"

    def read(self, filepath: Path, expected_fs: float | None, min_fs: float) -> LoadedSignal:
        if expected_fs is None:
            raise ValueError(
                "MatReader: .mat dosyalari icinde ornekleme hizi (fs) saklanmaz "
                "(orn. CWRU veri setinde). expected_fs zorunludur."
            )
        _validate_fs(expected_fs, expected_fs, min_fs)

        raw_dict, is_v73 = self._load_raw(filepath)
        channels, rpm = self._discover_channels(raw_dict, is_v73)

        if not channels:
            available = list(raw_dict.keys())
            raise ValueError(
                f"'{filepath}' icinde DE_time/FE_time/BA_time anahtari bulunamadi. "
                f"Dosyadaki anahtarlar: {available}"
            )

        cleaned = {name: _clean(arr) for name, arr in channels.items()}
        return LoadedSignal(
            channels=cleaned,
            fs=float(expected_fs),
            rpm=rpm,
            source_path=str(filepath),
            meta={"matched_keys": list(channels.keys()), "v7_3": is_v73},
        )

    @staticmethod
    def _load_raw(filepath: Path):
        """v5/v7 icin loadmat, v7.3 (HDF5) icin h5py fallback."""
        try:
            raw = loadmat(filepath)
            return {k: v for k, v in raw.items() if not k.startswith("__")}, False
        except (NotImplementedError, ValueError) as e:
            if not _HAS_H5PY:
                raise ImportError(
                    "Bu .mat dosyasi MATLAB v7.3 (HDF5) formatinda ve scipy.io.loadmat "
                    "bunu okuyamiyor. `pip install h5py` ile v7.3 destegini ekleyin."
                ) from e
            raw = {}
            with h5py.File(filepath, "r") as f:
                for key in f.keys():
                    raw[key] = np.array(f[key]).squeeze()
            return raw, True

    @staticmethod
    def _discover_channels(raw_dict: dict, is_v73: bool):
        """CWRU adlandirma sozlesmesine gore DE/FE/BA/RPM anahtarlarini bulur."""
        channels: dict[str, np.ndarray] = {}
        rpm = None

        for key, value in raw_dict.items():
            match = _CHANNEL_PATTERN.search(key)
            if match:
                role = match.group(1)  # "DE" | "FE" | "BA"
                arr = np.asarray(value).squeeze()
                if arr.ndim == 1 and arr.size > 1:
                    channels[role] = arr
                continue

            if _RPM_PATTERN.search(key):
                arr = np.asarray(value).squeeze()
                rpm = float(arr.item()) if arr.ndim == 0 or arr.size == 1 else float(np.mean(arr))

        return channels, rpm


class CsvReader(SignalReader):
    def can_read(self, filepath: Path) -> bool:
        return filepath.suffix.lower() == ".csv"

    def read(self, filepath: Path, expected_fs: float | None, min_fs: float) -> LoadedSignal:
        data = np.genfromtxt(filepath, delimiter=",", names=True)

        fs = expected_fs
        signal_col = None

        if data.dtype.names and len(data.dtype.names) >= 2 and any(
            "time" in n.lower() or n.lower() in ("t", "sn") for n in data.dtype.names
        ):
            time_col_name = next(
                n for n in data.dtype.names if "time" in n.lower() or n.lower() in ("t", "sn")
            )
            t = data[time_col_name].astype(np.float64)
            dt = np.median(np.diff(t))
            inferred_fs = 1.0 / dt
            if expected_fs is not None:
                _validate_fs(inferred_fs, expected_fs, min_fs)
                fs = expected_fs
            else:
                fs = inferred_fs
            other_cols = [n for n in data.dtype.names if n != time_col_name]
            signal_col = other_cols[0] if other_cols else None
            raw_signal = data[signal_col].astype(np.float64) if signal_col else t
        else:
            if fs is None:
                raise ValueError(
                    "CsvReader: dosyada zaman sutunu yok ve expected_fs verilmedi; "
                    "fs belirlenemiyor."
                )
            _validate_fs(fs, expected_fs, min_fs)
            if data.dtype.names:
                raw_signal = data[data.dtype.names[0]].astype(np.float64)
            else:
                raw_signal = np.asarray(data, dtype=np.float64).squeeze()

        signal = _clean(raw_signal)
        return LoadedSignal(
            channels={"main": signal},
            fs=float(fs),
            source_path=str(filepath),
        )

class SignalReaderFactory:
    """Dosya uzantisina gore uygun reader'a yonlendiren tek giris noktasi."""

    _readers: list[SignalReader] = [WavReader(), MatReader(), CsvReader()]

    @classmethod
    def load(cls, filepath: str, expected_fs: float | None = None,
            min_fs: float = 1000.0) -> LoadedSignal:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadi: {path}")

        for reader in cls._readers:
            if reader.can_read(path):
                return reader.read(path, expected_fs, min_fs)

        raise ValueError(
            f"Desteklenmeyen dosya formati: '{path.suffix}'. "
            f"Desteklenen formatlar: .wav, .mat, .csv"
        )


def load_signal_streaming(filepath: str, fs: float | None = None,
                        chunk_seconds: float = 1.0,
                        min_fs: float = 1000.0) -> Iterator[np.ndarray]:
    """Buyuk sinyal dosyalarini tek seferde tamamen bellege yuklemeden,
    sabit sureli (chunk_seconds) parcalar halinde okur.

    - .wav: scipy.io.wavfile.read(..., mmap=True) ile diskten bellek-esleme
    (mmap) uzerinden okur; sadece istenen parca fiilen belleğe alinir.
    - .csv: pandas.read_csv(..., chunksize=...) ile satir bazli parcali okur.
    - .mat (v7.3/HDF5): h5py dataset'i lazy-slicing ile parca parca okur
    (v5/v7 .mat formatlarinda tum degisken zaten tek blok halinde
    saklandigindan gercek parcali okuma mumkun degil; bu durumda
    ImportError/ValueError yerine acik bir NotImplementedError verilir).

    Her parca, penceleleme/FFT'den ONCE ayri ayri detrend(linear) edilir
    (`_clean` ile ayni kural) — boylece streaming modda da DC/trend
    kaldirma garantiye alinir; sadece dosya sonunda tek seferlik degil.

    Yields
    ------
    np.ndarray
        DC/trend'i kaldirilmis sinyal parcasi.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {path}")
    suffix = path.suffix.lower()

    if suffix == ".wav":
        file_fs, mm = wavfile.read(path, mmap=True)
        _validate_fs(file_fs, fs, min_fs)
        chunk_size = max(1, int(chunk_seconds * file_fs))
        n = len(mm)
        for start in range(0, n, chunk_size):
            chunk = np.asarray(mm[start:start + chunk_size], dtype=np.float64)
            if chunk.size < 2:
                continue
            yield detrend(chunk, type="linear")

    elif suffix == ".csv":
        if fs is None:
            raise ValueError(
                "load_signal_streaming: .csv icin fs (ornekleme hizi) zorunlu "
                "(dosyada saklanmiyor)."
            )
        _validate_fs(fs, fs, min_fs)
        chunk_rows = max(1, int(chunk_seconds * fs))
        for df_chunk in pd.read_csv(path, chunksize=chunk_rows):
            numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
            value_cols = [c for c in numeric_cols if "time" not in c.lower()]
            col = value_cols[0] if value_cols else numeric_cols[0]
            arr = df_chunk[col].to_numpy(dtype=np.float64)
            if arr.size < 2:
                continue
            yield detrend(arr, type="linear")

    elif suffix == ".mat":
        if fs is None:
            raise ValueError(
                "load_signal_streaming: .mat icin fs (ornekleme hizi) zorunlu "
                "(dosyada saklanmiyor)."
            )
        if not _HAS_H5PY:
            raise ImportError("load_signal_streaming: .mat v7.3 parcali okuma icin h5py gerekli.")
        _validate_fs(fs, fs, min_fs)
        chunk_size = max(1, int(chunk_seconds * fs))

        with h5py.File(path, "r") as f:
            target_key = next(
                (k for k in f.keys() if _CHANNEL_PATTERN.search(k)), None
            )
            if target_key is None:
                raise ValueError(
                    f"'{path}' icinde DE_time/FE_time/BA_time anahtari bulunamadi "
                    "(v5/v7 .mat dosyalari icin parcali okuma desteklenmiyor, "
                    "SignalReaderFactory.load kullanin)."
                )
            dataset = f[target_key]
            n = dataset.shape[0]
            for start in range(0, n, chunk_size):
                chunk = np.asarray(dataset[start:start + chunk_size]).squeeze().astype(np.float64)
                if chunk.size < 2:
                    continue
                yield detrend(chunk, type="linear")

    else:
        raise ValueError(
            f"load_signal_streaming: desteklenmeyen format '{suffix}'. "
            f"Desteklenen: .wav, .csv, .mat (v7.3)"
        )

def load_signal(filepath: str, expected_fs: float | None, min_fs: float = 1000.0):
    """Eski tekil-kanal API. signal_processing.load_signal ile ayni sozlesme;
    icten SignalReaderFactory'yi kullanir. Coklu kanal (.mat) durumunda
    'main' anahtarina en olasi ana kanali (DE varsa DE) atar."""
    loaded = SignalReaderFactory.load(filepath, expected_fs, min_fs)
    if "main" in loaded.channels:
        return loaded.channels["main"], loaded.fs
    preferred = next((k for k in ("DE", "FE", "BA") if k in loaded.channels), None)
    if preferred is None:
        preferred = next(iter(loaded.channels))
    return loaded.channels[preferred], loaded.fs

