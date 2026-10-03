"""CPU-friendly speech enhancement functions.

The enhancement methods in this module operate on short-time Fourier
transforms (STFTs) where appropriate so that noise estimation and
suppression are performed in the frequency domain.

The main design goals are:

- deterministic, CPU-friendly processing;
- preservation of speech-bearing frequency bins;
- configurable suppression rather than an aggressive hard gate;
- no mandatory ML dependency for the core DSP methods;
- compatibility with the existing CallClear API.

For an observed STFT magnitude |Y(k, m)| and an estimated noise
magnitude N(k), spectral subtraction estimates clean speech magnitude as:

    |X_hat(k, m)| = max(|Y(k, m)| - alpha * N(k), floor * |Y(k, m)|)

where alpha controls subtraction strength and ``floor`` prevents complete
suppression of a frequency bin.

For Wiener filtering, using an estimated speech-to-noise power ratio:

    SNR_hat(k, m) = max(P_Y(k, m) - P_N(k), 0) / (P_N(k) + eps)

the Wiener gain is:

    G(k, m) = SNR_hat(k, m) / (SNR_hat(k, m) + 1)

and the enhanced spectrum is:

    X_hat(k, m) = G(k, m) * Y(k, m)

The implementation deliberately keeps a spectral floor so that the
enhancer does not behave like an overly aggressive binary noise gate.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal


DEFAULT_NPERSEG = 256
DEFAULT_NOVERLAP = 192
DEFAULT_NOISE_PERCENTILE = 20.0
DEFAULT_SPECTRAL_SUBTRACTION_STRENGTH = 1.0
DEFAULT_SPECTRAL_FLOOR = 0.08
DEFAULT_WIENER_FLOOR = 0.08
DEFAULT_HIGH_PASS_CUTOFF_HZ = 120.0


def high_pass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = DEFAULT_HIGH_PASS_CUTOFF_HZ,
) -> np.ndarray:
    """Remove low-frequency rumble with a zero-phase Butterworth filter."""
    if not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("High-pass cutoff must be below Nyquist.")

    x = np.asarray(audio, dtype=np.float32)

    if x.size < 16:
        return x.copy()

    sos = signal.butter(
        4,
        cutoff_hz,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )

    # sosfiltfilt requires enough samples for its padding strategy.
    # Very short buffers are returned unchanged rather than failing.
    try:
        filtered = signal.sosfiltfilt(sos, x)
    except ValueError:
        return x.copy()

    return np.nan_to_num(filtered).astype(np.float32)


def _stft(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return an STFT using a size appropriate for speech at sample_rate."""
    del sample_rate

    x = np.asarray(audio, dtype=np.float32)

    if x.size == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.empty((0, 0), dtype=np.complex64),
            DEFAULT_NPERSEG,
            DEFAULT_NOVERLAP,
        )

    nperseg = min(DEFAULT_NPERSEG, x.size)

    # scipy requires noverlap < nperseg.
    if nperseg <= 1:
        noverlap = 0
    else:
        noverlap = min(DEFAULT_NOVERLAP, nperseg - 1)

    frequencies, times, spectrum = signal.stft(
        x,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )

    return frequencies, times, spectrum, nperseg, noverlap


def _istft(
    spectrum: np.ndarray,
    sample_count: int,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    """Reconstruct an audio signal from an STFT and restore its length."""
    if sample_count == 0:
        return np.empty(0, dtype=np.float32)

    _, reconstructed = signal.istft(
        spectrum,
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
    )

    reconstructed = np.asarray(reconstructed, dtype=np.float32)

    if reconstructed.size < sample_count:
        reconstructed = np.pad(
            reconstructed,
            (0, sample_count - reconstructed.size),
        )

    return np.nan_to_num(reconstructed[:sample_count]).astype(np.float32)


def _estimate_noise_power(
    magnitude: np.ndarray,
    percentile: float = DEFAULT_NOISE_PERCENTILE,
) -> np.ndarray:
    """Estimate per-frequency noise power from low-energy STFT frames.

    The lowest-energy frames are treated as the most likely non-speech
    frames. This is deliberately simple and deterministic, avoiding a
    second ML model in the core DSP path.
    """
    if magnitude.size == 0:
        return np.empty((0,), dtype=np.float32)

    if not 0.0 < percentile <= 100.0:
        raise ValueError("Noise percentile must be in the range (0, 100].")

    frame_energy = np.mean(magnitude**2, axis=0)

    frame_threshold = float(np.percentile(frame_energy, percentile))
    noise_frames = magnitude[:, frame_energy <= frame_threshold]

    if noise_frames.size == 0:
        noise_frames = magnitude

    noise_power = np.mean(noise_frames**2, axis=1)
    return np.maximum(noise_power, 1e-10).astype(np.float32)


def spectral_subtraction_filter(
    audio: np.ndarray,
    sample_rate: int,
    subtraction_strength: float = DEFAULT_SPECTRAL_SUBTRACTION_STRENGTH,
    spectral_floor: float = DEFAULT_SPECTRAL_FLOOR,
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE,
) -> np.ndarray:
    """Suppress estimated stationary noise using STFT spectral subtraction.

    The estimated noise spectrum is derived from the lowest-energy
    percentage of STFT frames. A soft spectral floor prevents complete
    removal of frequency bins that may contain speech.
    """
    if subtraction_strength < 0.0:
        raise ValueError("Subtraction strength must be non-negative.")

    if not 0.0 <= spectral_floor <= 1.0:
        raise ValueError("Spectral floor must be between 0 and 1.")

    _, _, spectrum, nperseg, noverlap = _stft(audio, sample_rate)

    if spectrum.size == 0:
        return np.asarray(audio, dtype=np.float32).copy()

    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    noise_power = _estimate_noise_power(
        magnitude,
        percentile=noise_percentile,
    )
    noise_magnitude = np.sqrt(noise_power)[:, np.newaxis]

    cleaned_magnitude = magnitude - subtraction_strength * noise_magnitude

    floor_magnitude = spectral_floor * magnitude
    cleaned_magnitude = np.maximum(cleaned_magnitude, floor_magnitude)

    enhanced_spectrum = cleaned_magnitude * np.exp(1j * phase)

    return _istft(
        enhanced_spectrum,
        len(np.asarray(audio)),
        nperseg,
        noverlap,
    )


def wiener_filter(
    audio: np.ndarray,
    sample_rate: int,
    spectral_floor: float = DEFAULT_WIENER_FLOOR,
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE,
) -> np.ndarray:
    """Apply STFT Wiener filtering using an estimated noise spectrum.

    The Wiener gain approaches zero where estimated noise dominates and
    approaches one where estimated speech power dominates.

    A configurable gain floor prevents complete removal of speech-bearing
    bins and makes the method less aggressive than a hard spectral gate.
    """
    if not 0.0 <= spectral_floor <= 1.0:
        raise ValueError("Spectral floor must be between 0 and 1.")

    _, _, spectrum, nperseg, noverlap = _stft(audio, sample_rate)

    if spectrum.size == 0:
        return np.asarray(audio, dtype=np.float32).copy()

    magnitude = np.abs(spectrum)
    power = magnitude**2

    noise_power = _estimate_noise_power(
        magnitude,
        percentile=noise_percentile,
    )[:, np.newaxis]

    speech_power = np.maximum(power - noise_power, 0.0)

    snr_estimate = speech_power / (noise_power + 1e-10)
    gain = snr_estimate / (snr_estimate + 1.0)

    gain = np.maximum(gain, spectral_floor)

    enhanced_spectrum = spectrum * gain

    return _istft(
        enhanced_spectrum,
        len(np.asarray(audio)),
        nperseg,
        noverlap,
    )


def spectral_gate(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -18.0,
) -> np.ndarray:
    """Apply a conservative soft spectral gate.

    This method is retained for API compatibility. Unlike the original
    implementation, the default threshold is intentionally less aggressive
    and uses a soft mask rather than a binary magnitude gate.
    """
    if threshold_db >= 0.0:
        raise ValueError("threshold_db must be negative.")

    _, _, spectrum, nperseg, noverlap = _stft(audio, sample_rate)

    if spectrum.size == 0:
        return np.asarray(audio, dtype=np.float32).copy()

    magnitude = np.abs(spectrum)

    noise_power = _estimate_noise_power(magnitude)
    noise_magnitude = np.sqrt(noise_power)[:, np.newaxis]

    threshold_ratio = 10.0 ** (threshold_db / 20.0)
    threshold = noise_magnitude / max(threshold_ratio, 1e-6)

    ratio = magnitude / (threshold + 1e-10)

    # Soft mask:
    # ratio << 1  -> suppression
    # ratio >= 1  -> approximately unchanged
    mask = np.clip(ratio, 0.08, 1.0)

    enhanced_spectrum = spectrum * mask

    return _istft(
        enhanced_spectrum,
        len(np.asarray(audio)),
        nperseg,
        noverlap,
    )


@lru_cache(maxsize=1)
def _load_silero_vad_model():
    """Load and cache the Silero VAD model."""
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD is not installed. Install with: "
            "uv sync --extra asr"
        ) from exc

    return load_silero_vad()


def silero_vad_trim(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim leading/trailing non-speech using cached Silero VAD."""
    try:
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD is not installed. Install with: "
            "uv sync --extra asr"
        ) from exc

    if sample_rate not in (8000, 16000):
        raise ValueError(
            "Silero VAD supports 8 kHz and 16 kHz input in this project."
        )

    import torch

    x = np.asarray(audio, dtype=np.float32)

    if x.size == 0:
        return x.copy()

    model = _load_silero_vad_model()
    tensor = torch.from_numpy(x)

    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
    )

    if not timestamps:
        return x.copy()

    start = int(timestamps[0]["start"])
    end = int(timestamps[-1]["end"])

    return np.asarray(x[start:end], dtype=np.float32)


def enhance_audio(
    audio: np.ndarray,
    sample_rate: int,
    method: str = "wiener",
) -> np.ndarray:
    """Run high-pass filtering followed by the selected enhancement method.

    Supported methods:

    - ``highpass``: only the 120 Hz high-pass filter;
    - ``spectral_gate``: conservative soft STFT spectral gate;
    - ``wiener``: STFT Wiener enhancement;
    - ``vad``: Silero VAD leading/trailing silence trimming.
    """
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
        raise ValueError(
            "method must be one of: wiener, spectral_gate, vad, highpass"
        )

    peak = float(np.max(np.abs(x))) if x.size else 0.0

    if peak > 0.99:
        x = x / peak * 0.99

    return np.asarray(x, dtype=np.float32)