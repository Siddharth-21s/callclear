"""CPU-friendly speech enhancement functions."""

import numpy as np
from scipy import signal


def high_pass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 120.0) -> np.ndarray:
    """Remove low-frequency rumble with a zero-phase Butterworth filter."""
    if not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("High-pass cutoff must be below Nyquist.")
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def wiener_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply a conservative SciPy Wiener filter."""
    del sample_rate
    filtered = signal.wiener(np.asarray(audio, dtype=np.float32), mysize=31)
    return np.nan_to_num(filtered).astype(np.float32)


def spectral_gate(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -35.0,
) -> np.ndarray:
    """Apply a simple magnitude spectral gate without external ML dependencies."""
    del sample_rate
    frequencies, times, zxx = signal.stft(
        audio,
        nperseg=512,
        noverlap=384,
        boundary="zeros",
    )
    magnitude = np.abs(zxx)
    reference = np.percentile(magnitude, 20.0, axis=1, keepdims=True)
    reference = np.maximum(reference, 1e-8)
    threshold = reference * (10.0 ** (-threshold_db / 20.0))
    mask = magnitude >= threshold
    gated = zxx * mask
    _, reconstructed = signal.istft(
        gated,
        nperseg=512,
        noverlap=384,
        input_onesided=True,
    )
    return reconstructed[: len(audio)].astype(np.float32)


def silero_vad_trim(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim leading/trailing non-speech using Silero VAD when installed."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD is not installed. Install with: "
            "uv sync --extra asr"
        ) from exc
    if sample_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports 8 kHz and 16 kHz input in this project.")
    import torch

    model = load_silero_vad()
    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=sample_rate)
    if not timestamps:
        return np.asarray(audio, dtype=np.float32)
    start = int(timestamps[0]["start"])
    end = int(timestamps[-1]["end"])
    return np.asarray(audio[start:end], dtype=np.float32)


def enhance_audio(audio: np.ndarray, sample_rate: int, method: str = "wiener") -> np.ndarray:
    """Run high-pass filtering followed by a selected enhancement method."""
    x = high_pass_filter(audio, sample_rate)
    if method == "wiener":
        x = wiener_filter(x, sample_rate)
    elif method == "spectral_gate":
        x = spectral_gate(x, sample_rate)
    elif method == "vad":
        x = silero_vad_trim(x, sample_rate)
    elif method == "highpass":
        pass
    else:
        raise ValueError("method must be one of: wiener, spectral_gate, vad, highpass")
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.99:
        x = x / peak * 0.99
    return x.astype(np.float32)
