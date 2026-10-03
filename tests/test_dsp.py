"""Offline DSP tests."""

import numpy as np

from app.benchmarks.metrics import percentile
from app.dsp.enhancement import enhance_audio, high_pass_filter
from app.dsp.telephony import (
    DegradationConfig,
    add_noise_at_snr,
    degrade_audio,
    mu_law_decode,
    mu_law_encode,
)


def test_mu_law_round_trip_is_bounded() -> None:
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    y = mu_law_decode(mu_law_encode(x))
    assert np.max(np.abs(y)) <= 1.0
    assert np.mean(np.abs(y - x)) < 0.01


def test_degrade_outputs_8khz() -> None:
    source = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    output, rate = degrade_audio(source, 16000, DegradationConfig(snr_db=10, seed=1))
    assert rate == 8000
    assert output.dtype == np.float32
    assert output.size == 8000


def test_noise_changes_signal() -> None:
    x = np.ones(8000, dtype=np.float32) * 0.2
    y = add_noise_at_snr(x, 0, seed=2)
    assert not np.array_equal(x, y)


def test_high_pass_removes_dc() -> None:
    x = np.ones(8000, dtype=np.float32) * 0.5
    y = high_pass_filter(x, 8000)
    assert abs(float(np.mean(y))) < 0.05


def test_enhancement_returns_audio() -> None:
    x = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000).astype(np.float32)
    y = enhance_audio(x, 8000, method="highpass")
    assert y.size == x.size


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
