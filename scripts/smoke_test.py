"""Offline end-to-end smoke test using synthetic speech-like audio."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmarks.metrics import summarize_latency  # noqa: E402
from app.dsp.enhancement import enhance_audio  # noqa: E402
from app.dsp.telephony import DegradationConfig, degrade_audio  # noqa: E402


def main() -> None:
    """Run a small offline pipeline check."""
    sample_rate = 16_000
    duration = 1.0
    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False,
    )

    audio = (
        0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.05 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)

    config = DegradationConfig(
        snr_db=10.0,
        seed=7,
    )

    degraded, rate = degrade_audio(
        audio,
        sample_rate,
        config,
    )

    enhanced = enhance_audio(
        degraded,
        rate,
        method="wiener",
    )

    output = ROOT / "results"
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    degraded_path = output / "smoke_degraded.wav"
    enhanced_path = output / "smoke_enhanced.wav"

    sf.write(
        degraded_path,
        degraded,
        rate,
        subtype="PCM_16",
    )

    sf.write(
        enhanced_path,
        enhanced,
        rate,
        subtype="PCM_16",
    )

    latency_values = [0.1, 0.2, 0.3]
    latency = summarize_latency(latency_values)

    print("Smoke test passed.")
    print(f"Degraded audio: {degraded_path}")
    print(f"Enhanced audio: {enhanced_path}")
    print(f"Latency summary: {latency}")


if __name__ == "__main__":
    main()