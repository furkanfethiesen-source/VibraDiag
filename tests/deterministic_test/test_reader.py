"""
Unit tests for deterministic_tools.reader
==========================================
Tests polymorphic readers (WAV, MAT, CSV), validation, detrending,
streaming generation, and error conditions without external datasets.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from scipy.io import loadmat, savemat, wavfile

from src.deterministic_tools.reader import (
    CsvReader,
    LoadedSignal,
    MatReader,
    SignalReaderFactory,
    WavReader,
    _clean,
    _validate_fs,
    _validate_signal_array,
    load_signal_streaming,
)


class TestLoadedSignalAndValidations:
    def test_loaded_signal_success(self):
        arr = np.array([1.0, 2.0, -1.0, 0.5])
        sig = LoadedSignal(channels={"main": arr}, fs=12000.0, rpm=1800.0)
        assert "main" in sig.channels
        assert sig.fs == 12000.0
        assert sig.rpm == 1800.0

    def test_validation_empty_or_all_zeros(self):
        with pytest.raises(ValueError, match="bos veya tamamen sifir"):
            _validate_signal_array(np.array([]))
        with pytest.raises(ValueError, match="bos veya tamamen sifir"):
            _validate_signal_array(np.zeros(100))

    def test_validation_nan_or_inf(self):
        with pytest.raises(ValueError, match="NaN veya Inf"):
            _validate_signal_array(np.array([1.0, np.nan, 2.0]))
        with pytest.raises(ValueError, match="NaN veya Inf"):
            _validate_signal_array(np.array([1.0, np.inf, 2.0]))

    def test_validate_fs_bounds(self):
        with pytest.raises(ValueError, match="Ornekleme hizi cok dusuk"):
            _validate_fs(fs=500.0, expected_fs=None, min_fs=1000.0)

        with pytest.raises(ValueError, match="Beklenmeyen ornekleme hizi"):
            _validate_fs(fs=10000.0, expected_fs=12000.0, min_fs=1000.0)

        # Tolerance match should pass
        _validate_fs(fs=12000.0, expected_fs=12000.0, min_fs=1000.0)
        _validate_fs(fs=12050.0, expected_fs=12000.0, min_fs=1000.0)  # within 1%

    def test_clean_detrending(self):
        t = np.linspace(0, 1, 1000)
        # linear trend + DC offset
        raw = 5.0 + 2.0 * t + np.sin(2 * np.pi * 10 * t)
        cleaned = _clean(raw)
        assert np.isclose(np.mean(cleaned), 0.0, atol=1e-2)
        assert cleaned.shape == (1000,)


class TestWavReader:
    def test_wav_read_success(self, tmp_path):
        fs = 8000
        duration = 0.2
        t = np.arange(0, duration, 1.0 / fs)
        sine = (0.5 * np.sin(2 * np.pi * 100 * t) * 32767).astype(np.int16)

        wav_file = tmp_path / "test_signal.wav"
        wavfile.write(wav_file, fs, sine)

        reader = WavReader()
        assert reader.can_read(wav_file)
        assert not reader.can_read(tmp_path / "test.csv")

        loaded = reader.read(wav_file, expected_fs=8000.0, min_fs=1000.0)
        assert "main" in loaded.channels
        assert loaded.fs == 8000.0
        assert len(loaded.channels["main"]) == len(sine)

    def test_wav_read_mismatched_fs(self, tmp_path):
        fs = 4000
        wav_file = tmp_path / "low_fs.wav"
        wavfile.write(wav_file, fs, np.array([100, -100, 50, -50], dtype=np.int16))

        reader = WavReader()
        with pytest.raises(ValueError, match="Beklenmeyen ornekleme hizi"):
            reader.read(wav_file, expected_fs=12000.0, min_fs=1000.0)


class TestMatReader:
    def test_mat_read_cwru_style_success(self, tmp_path):
        fs = 12000.0
        n_samples = 500
        t = np.arange(n_samples) / fs
        de_data = np.sin(2 * np.pi * 100 * t).reshape(-1, 1)
        fe_data = (0.5 * np.cos(2 * np.pi * 100 * t)).reshape(-1, 1)

        mat_dict = {
            "X105_DE_time": de_data,
            "X105_FE_time": fe_data,
            "X105_RPM": np.array([1797.0]),
        }
        mat_file = tmp_path / "105.mat"
        savemat(mat_file, mat_dict)

        reader = MatReader()
        assert reader.can_read(mat_file)
        assert not reader.can_read(tmp_path / "105.wav")

        loaded = reader.read(mat_file, expected_fs=12000.0, min_fs=1000.0)
        assert "DE" in loaded.channels
        assert "FE" in loaded.channels
        assert loaded.rpm == 1797.0
        assert loaded.fs == 12000.0
        assert loaded.meta.get("v7_3") is False

    def test_mat_read_missing_expected_fs(self, tmp_path):
        mat_file = tmp_path / "empty.mat"
        savemat(mat_file, {"data": [1, 2, 3]})
        reader = MatReader()
        with pytest.raises(ValueError, match="expected_fs zorunludur"):
            reader.read(mat_file, expected_fs=None, min_fs=1000.0)

    def test_mat_read_no_known_channels(self, tmp_path):
        mat_file = tmp_path / "unknown_keys.mat"
        savemat(mat_file, {"some_random_key": np.array([1.0, 2.0, 3.0])})
        reader = MatReader()
        with pytest.raises(ValueError, match="DE_time/FE_time/BA_time anahtari bulunamadi"):
            reader.read(mat_file, expected_fs=12000.0, min_fs=1000.0)


class TestCsvReader:
    def test_csv_with_time_column(self, tmp_path):
        csv_file = tmp_path / "signal_with_time.csv"
        csv_file.write_text(
            "time,vibration\n"
            "0.0000,0.12\n"
            "0.0001,0.45\n"
            "0.0002,-0.30\n"
            "0.0003,0.15\n"
            "0.0004,0.02\n"
        )
        reader = CsvReader()
        assert reader.can_read(csv_file)
        loaded = reader.read(csv_file, expected_fs=None, min_fs=1000.0)
        assert "main" in loaded.channels
        assert np.isclose(loaded.fs, 10000.0, atol=10.0)

    def test_csv_single_column_without_time(self, tmp_path):
        csv_file = tmp_path / "single_col.csv"
        csv_file.write_text("vibration\n1.2\n-0.8\n0.5\n0.3\n-0.2\n")
        reader = CsvReader()
        # Without expected_fs, it should fail
        with pytest.raises(ValueError, match="zaman sutunu yok ve expected_fs verilmedi"):
            reader.read(csv_file, expected_fs=None, min_fs=1000.0)

        # With expected_fs, it succeeds
        loaded = reader.read(csv_file, expected_fs=12000.0, min_fs=1000.0)
        assert loaded.fs == 12000.0
        assert len(loaded.channels["main"]) == 5


class TestSignalReaderFactory:
    def test_factory_load_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("1, 2, 3")
        with pytest.raises(ValueError, match="Desteklenmeyen dosya formati"):
            SignalReaderFactory.load(str(txt_file))

    def test_factory_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Dosya bulunamadi"):
            SignalReaderFactory.load("non_existent_file.mat")

    def test_factory_load_dispatch_wav(self, tmp_path):
        fs = 8000
        wav_file = tmp_path / "factory_test.wav"
        wavfile.write(wav_file, fs, np.array([1000, -1000, 500, -500], dtype=np.int16))
        loaded = SignalReaderFactory.load(str(wav_file), expected_fs=8000.0)
        assert isinstance(loaded, LoadedSignal)
        assert loaded.fs == 8000.0


class TestStreamingLoader:
    def test_streaming_csv(self, tmp_path):
        csv_file = tmp_path / "stream_test.csv"
        lines = ["time,val"] + [f"{i*0.001:.3f},{np.sin(i):.4f}" for i in range(2500)]
        csv_file.write_text("\n".join(lines))

        # 1000 Hz, chunk_seconds = 1.0 -> chunks of 1000 samples
        chunks = list(load_signal_streaming(str(csv_file), fs=1000.0, chunk_seconds=1.0, min_fs=500.0))
        assert len(chunks) == 3  # 1000, 1000, 500
        assert len(chunks[0]) == 1000
        assert len(chunks[2]) == 500

    def test_streaming_wav(self, tmp_path):
        fs = 4000
        duration = 0.5
        t = np.arange(0, duration, 1.0 / fs)
        sine = (0.5 * np.sin(2 * np.pi * 50 * t) * 32767).astype(np.int16)
        wav_file = tmp_path / "stream.wav"
        wavfile.write(wav_file, fs, sine)

        chunks = list(load_signal_streaming(str(wav_file), fs=4000.0, chunk_seconds=0.2, min_fs=1000.0))
        assert len(chunks) >= 2
        assert all(len(c) > 0 for c in chunks)
