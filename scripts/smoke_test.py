"""Offline end-to-end smoke test using synthetic speech-like audio."""

from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmarks.metrics import summarize_latency
from app.dsp.enhancement import enhance_audio
from app.dsp.telephony import DegradationConfig, degrade_audio


def main() -> None:
    """Run the smoke test."""
    sample_rate = 16000
    duration = 2.0
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    audio = (
        0.25 * np.sin(2 * np.pi * 220 * t)
        + 0.12 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)
    degraded, rate = degrade_audio(audio, sample_rate, DegradationConfig(snr_db=10.0, seed=7))
    enhanced = enhance_audio(degraded, rate, method="wiener")
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    sf.write(output / "smoke_degraded.wav", degraded, rate)
    sf.write(output / "smoke_enhanced.wav", enhanced, rate)
    metrics = summarize_latency([0.10, 0.20, 0.30])
    assert rate == 8000
    assert len(degraded) > 0
    assert len(enhanced) > 0
    assert metrics["p50"] == 0.20
    print("PASS: synthetic degradation")
    print("PASS: DSP enhancement")
    print("PASS: latency metric")
    print("PASS: smoke test complete")


if __name__ == "__main__":
    main()
