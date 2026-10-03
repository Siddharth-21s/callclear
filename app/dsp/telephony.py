"""Telephone-channel degradation functions."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class DegradationConfig:
    """Parameters for the synthetic telephone channel."""

    target_sample_rate: int = 8000
    low_cut_hz: float = 300.0
    high_cut_hz: float = 3400.0
    snr_db: float = 10.0
    packet_loss_ms: float = 60.0
    packet_loss_probability: float = 0.02
    seed: int = 42


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Convert audio to float32 and peak-normalize when necessary."""
    samples = np.asarray(audio, dtype=np.float32)

    if samples.ndim != 1:
        raise ValueError("Only mono 1-D audio is supported.")

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0

    if peak > 1.0:
        samples = samples / peak

    return samples


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample audio using polyphase filtering."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")

    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32).copy()

    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd

    return signal.resample_poly(
        audio,
        up,
        down,
    ).astype(np.float32)


def band_limit(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Apply a zero-phase Butterworth telephone-band filter."""
    if not 0 < low_hz < high_hz < sample_rate / 2:
        raise ValueError("Band edges must lie strictly inside Nyquist.")

    sos = signal.butter(
        6,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def mu_law_encode(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """Encode normalized float audio as 8-bit μ-law codes."""
    if mu != 255:
        raise ValueError("G.711 μ-law requires mu=255.")

    x = np.clip(
        np.asarray(audio, dtype=np.float32),
        -1.0,
        1.0,
    )

    compressed = np.sign(x) * (
        np.log1p(mu * np.abs(x)) / np.log1p(mu)
    )

    encoded = np.round(
        ((compressed + 1.0) * 0.5) * 255.0
    ).astype(np.uint8)

    return encoded


def mu_law_decode(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """Decode 8-bit μ-law codes back to normalized float audio."""
    if mu != 255:
        raise ValueError("G.711 μ-law requires mu=255.")

    codes = np.asarray(audio, dtype=np.uint8)

    compressed = (
        (codes.astype(np.float32) / 255.0) * 2.0
    ) - 1.0

    decoded = np.sign(compressed) * (
        (1.0 + mu) ** np.abs(compressed) - 1.0
    ) / mu

    return decoded.astype(np.float32)


def add_noise_at_snr(
    audio: np.ndarray,
    snr_db: float,
    seed: int = 42,
) -> np.ndarray:
    """Add white Gaussian noise at the requested signal-to-noise ratio."""
    x = np.asarray(audio, dtype=np.float32)
    power = float(np.mean(np.square(x)))

    if power <= 1e-15:
        return x.copy()

    noise_power = power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0.0,
        np.sqrt(noise_power),
        size=x.shape,
    ).astype(np.float32)

    return (x + noise).astype(np.float32)


def apply_packet_loss(
    audio: np.ndarray,
    sample_rate: int,
    gap_ms: float = 60.0,
    probability: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Insert random packet-loss bursts into the audio."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Packet-loss probability must be between 0 and 1."
        )

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if gap_ms <= 0:
        raise ValueError("Packet-loss gap must be positive.")

    x = np.asarray(audio, dtype=np.float32).copy()

    if x.size == 0 or probability == 0.0:
        return x

    gap_samples = max(
        1,
        int(sample_rate * gap_ms / 1000.0),
    )

    rng = np.random.default_rng(seed)

    packet_count = int(np.ceil(x.size / gap_samples))
    loss_events = rng.random(packet_count) < probability

    for packet_index, lost in enumerate(loss_events):
        if not lost:
            continue

        start = packet_index * gap_samples
        end = min(start + gap_samples, x.size)
        x[start:end] = 0.0

    return x


def degrade_audio(
    audio: np.ndarray,
    source_rate: int,
    config: DegradationConfig,
) -> tuple[np.ndarray, int]:
    """Run the complete synthetic telephone channel."""
    x = normalize_audio(audio)

    x = resample_audio(
        x,
        source_rate,
        config.target_sample_rate,
    )

    x = band_limit(
        x,
        config.target_sample_rate,
        config.low_cut_hz,
        config.high_cut_hz,
    )

    encoded = mu_law_encode(x)
    x = mu_law_decode(encoded)

    x = add_noise_at_snr(
        x,
        config.snr_db,
        config.seed,
    )

    x = apply_packet_loss(
        x,
        config.target_sample_rate,
        config.packet_loss_ms,
        config.packet_loss_probability,
        config.seed + 1,
    )

    peak = float(np.max(np.abs(x))) if x.size else 0.0

    if peak > 0.99:
        x = x / peak * 0.99

    return x.astype(np.float32), config.target_sample_rate