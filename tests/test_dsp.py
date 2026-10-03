"""Offline DSP tests."""

import numpy as np

from app.benchmarks.metrics import percentile
from app.dsp.enhancement import (
    enhance_audio,
    high_pass_filter,
    spectral_gate,
    spectral_subtraction_filter,
    wiener_filter,
)
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
    source = np.sin(
        2 * np.pi * 440 * np.arange(16000) / 16000
    ).astype(np.float32)

    output, rate = degrade_audio(
        source,
        16000,
        DegradationConfig(snr_db=10, seed=1),
    )

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
    x = np.sin(
        2 * np.pi * 440 * np.arange(8000) / 8000
    ).astype(np.float32)

    y = enhance_audio(x, 8000, method="highpass")

    assert y.size == x.size
    assert y.dtype == np.float32


def test_wiener_filter_preserves_length() -> None:
    rng = np.random.default_rng(123)

    clean = 0.25 * np.sin(
        2 * np.pi * 440 * np.arange(8000) / 8000
    )

    noisy = (
        clean
        + rng.normal(0.0, 0.05, size=8000)
    ).astype(np.float32)

    enhanced = wiener_filter(noisy, 8000)

    assert enhanced.size == noisy.size
    assert enhanced.dtype == np.float32
    assert np.all(np.isfinite(enhanced))


def test_spectral_subtraction_preserves_length() -> None:
    rng = np.random.default_rng(456)

    clean = 0.25 * np.sin(
        2 * np.pi * 440 * np.arange(8000) / 8000
    )

    noisy = (
        clean
        + rng.normal(0.0, 0.05, size=8000)
    ).astype(np.float32)

    enhanced = spectral_subtraction_filter(noisy, 8000)

    assert enhanced.size == noisy.size
    assert enhanced.dtype == np.float32
    assert np.all(np.isfinite(enhanced))


def test_spectral_gate_preserves_length() -> None:
    x = (
        0.25
        * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
    ).astype(np.float32)

    y = spectral_gate(x, 8000)

    assert y.size == x.size
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


def test_enhancement_methods_are_deterministic() -> None:
    rng = np.random.default_rng(789)

    x = (
        0.2 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
        + rng.normal(0.0, 0.02, size=8000)
    ).astype(np.float32)

    for method in ("highpass", "spectral_gate", "wiener"):
        first = enhance_audio(x, 8000, method=method)
        second = enhance_audio(x, 8000, method=method)

        assert np.array_equal(first, second)


def test_enhancement_rejects_unknown_method() -> None:
    x = np.zeros(8000, dtype=np.float32)

    try:
        enhance_audio(x, 8000, method="unknown")
    except ValueError as exc:
        assert "method must be one of" in str(exc)
    else:
        raise AssertionError("Unknown enhancement method should fail.")


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5