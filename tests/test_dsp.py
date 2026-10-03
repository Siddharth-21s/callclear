"""Offline DSP and telephony tests."""

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
    apply_packet_loss,
    band_limit,
    degrade_audio,
    mu_law_decode,
    mu_law_encode,
)


def _measured_snr_db(
    clean: np.ndarray,
    noisy: np.ndarray,
) -> float:
    """Measure SNR between a clean signal and its noisy version."""
    signal_power = float(np.mean(np.square(clean)))
    noise = noisy - clean
    noise_power = float(np.mean(np.square(noise)))

    return 10.0 * np.log10(signal_power / noise_power)


def _longest_zero_run(audio: np.ndarray) -> int:
    """Return the longest consecutive run of exact zeros."""
    zero = np.asarray(audio) == 0.0

    longest = 0
    current = 0

    for value in zero:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def test_mu_law_round_trip_is_bounded() -> None:
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    y = mu_law_decode(mu_law_encode(x))

    assert np.max(np.abs(y)) <= 1.0
    assert np.mean(np.abs(y - x)) < 0.01


def test_mu_law_known_endpoint_codes() -> None:
    x = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float32,
    )

    encoded = mu_law_encode(x)

    assert encoded.tolist() == [0, 128, 255]


def test_mu_law_zero_decodes_near_zero() -> None:
    encoded = np.array([128], dtype=np.uint8)
    decoded = mu_law_decode(encoded)

    assert abs(float(decoded[0])) < 0.01


def test_degrade_outputs_8khz() -> None:
    source = np.sin(
        2 * np.pi * 440 * np.arange(16000) / 16000
    ).astype(np.float32)

    output, rate = degrade_audio(
        source,
        16000,
        DegradationConfig(
            snr_db=10,
            seed=1,
        ),
    )

    assert rate == 8000
    assert output.dtype == np.float32
    assert output.size == 8000


def test_noise_changes_signal() -> None:
    x = np.ones(8000, dtype=np.float32) * 0.2
    y = add_noise_at_snr(x, 0, seed=2)

    assert not np.array_equal(x, y)


def test_requested_snr_is_within_half_db() -> None:

    clean = (
        0.2
        * np.sin(
            2 * np.pi * 440 * np.arange(16000) / 16000
        )
    ).astype(np.float32)

    noisy = add_noise_at_snr(
        clean,
        10.0,
        seed=123,
    )

    measured = _measured_snr_db(clean, noisy)

    assert abs(measured - 10.0) <= 0.5


def test_band_limit_suppresses_energy_outside_telephone_band() -> None:
    sample_rate = 8000
    samples = np.arange(sample_rate, dtype=np.float32)

    in_band = np.sin(
        2 * np.pi * 1000 * samples / sample_rate
    )

    low_outside = np.sin(
        2 * np.pi * 100 * samples / sample_rate
    )

    high_outside = np.sin(
        2 * np.pi * 3800 * samples / sample_rate
    )

    source = (
        in_band
        + low_outside
        + high_outside
    ).astype(np.float32)

    filtered = band_limit(
        source,
        sample_rate,
        300.0,
        3400.0,
    )

    # Ignore filter transients at the edges.
    source_core = source[1000:-1000]
    filtered_core = filtered[1000:-1000]

    source_energy = float(np.mean(source_core**2))
    filtered_energy = float(np.mean(filtered_core**2))

    assert filtered_energy < source_energy


def test_band_limit_reduces_100hz_component_by_more_than_20db() -> None:
    sample_rate = 8000
    samples = np.arange(sample_rate, dtype=np.float32)

    low = np.sin(
        2 * np.pi * 100 * samples / sample_rate
    ).astype(np.float32)

    filtered = band_limit(
        low,
        sample_rate,
        300.0,
        3400.0,
    )

    core = slice(1000, -1000)

    input_rms = float(
        np.sqrt(np.mean(low[core] ** 2))
    )
    output_rms = float(
        np.sqrt(np.mean(filtered[core] ** 2))
    )

    attenuation_db = 20.0 * np.log10(
        max(output_rms, 1e-12) / input_rms
    )

    assert attenuation_db < -20.0


def test_packet_loss_probability_zero_preserves_audio() -> None:
    x = np.sin(
        2 * np.pi * 440 * np.arange(8000) / 8000
    ).astype(np.float32)

    y = apply_packet_loss(
        x,
        sample_rate=8000,
        gap_ms=60,
        probability=0.0,
        seed=42,
    )

    assert np.array_equal(x, y)


def test_packet_loss_creates_expected_gap_count_and_length() -> None:
    sample_rate = 8000
    gap_ms = 60.0
    probability = 0.25
    seed = 42

    x = np.ones(8000, dtype=np.float32)

    y = apply_packet_loss(
        x,
        sample_rate=sample_rate,
        gap_ms=gap_ms,
        probability=probability,
        seed=seed,
    )

    gap_samples = int(
        sample_rate * gap_ms / 1000.0
    )

    rng = np.random.default_rng(seed)
    packet_count = int(
        np.ceil(x.size / gap_samples)
    )
    expected_losses = (
        rng.random(packet_count) < probability
    )

    expected_loss_count = int(
        np.sum(expected_losses)
    )

    assert int(np.sum(y == 0.0)) == (
        expected_loss_count * gap_samples
    )

    assert _longest_zero_run(y) == gap_samples


def test_high_pass_removes_dc() -> None:
    x = np.ones(8000, dtype=np.float32) * 0.5
    y = high_pass_filter(x, 8000)

    assert abs(float(np.mean(y))) < 0.05


def test_enhancement_returns_audio() -> None:
    x = np.sin(
        2 * np.pi * 440 * np.arange(8000) / 8000
    ).astype(np.float32)

    y = enhance_audio(
        x,
        8000,
        method="highpass",
    )

    assert y.size == x.size
    assert y.dtype == np.float32


def test_wiener_filter_preserves_length() -> None:
    rng = np.random.default_rng(123)

    clean = 0.25 * np.sin(
        2 * np.pi * 440 * np.arange(8000) / 8000
    )

    noisy = (
        clean
        + rng.normal(
            0.0,
            0.05,
            size=8000,
        )
    ).astype(np.float32)

    enhanced = wiener_filter(
        noisy,
        8000,
    )

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
        + rng.normal(
            0.0,
            0.05,
            size=8000,
        )
    ).astype(np.float32)

    enhanced = spectral_subtraction_filter(
        noisy,
        8000,
    )

    assert enhanced.size == noisy.size
    assert enhanced.dtype == np.float32
    assert np.all(np.isfinite(enhanced))


def test_spectral_gate_preserves_length() -> None:
    x = (
        0.25
        * np.sin(
            2 * np.pi * 440 * np.arange(8000) / 8000
        )
    ).astype(np.float32)

    y = spectral_gate(
        x,
        8000,
    )

    assert y.size == x.size
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


def test_enhancement_methods_are_deterministic() -> None:
    rng = np.random.default_rng(789)

    x = (
        0.2
        * np.sin(
            2 * np.pi * 440 * np.arange(8000) / 8000
        )
        + rng.normal(
            0.0,
            0.02,
            size=8000,
        )
    ).astype(np.float32)

    for method in (
        "highpass",
        "spectral_gate",
        "wiener",
    ):
        first = enhance_audio(
            x,
            8000,
            method=method,
        )

        second = enhance_audio(
            x,
            8000,
            method=method,
        )

        assert np.array_equal(
            first,
            second,
        )


def test_enhancement_rejects_unknown_method() -> None:
    x = np.zeros(
        8000,
        dtype=np.float32,
    )

    try:
        enhance_audio(
            x,
            8000,
            method="unknown",
        )
    except ValueError as exc:
        assert "method must be one of" in str(exc)
    else:
        raise AssertionError(
            "Unknown enhancement method should fail."
        )


def test_percentile() -> None:
    assert percentile(
        [1, 2, 3, 4],
        50,
    ) == 2.5