"""Local benchmark runner for cached Hindi FLEURS audio."""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import soundfile as sf

from app.asr.engine import WhisperEngine
from app.benchmarks.metrics import percentile
from app.dsp.enhancement import enhance_audio
from app.dsp.telephony import DegradationConfig, degrade_audio


def _write_wav(
    path: Path,
    samples,
    sample_rate: int,
) -> None:
    """Write floating-point audio as a PCM-16 WAV file."""
    sf.write(
        path,
        samples,
        sample_rate,
        subtype="PCM_16",
    )


def _transcribe_and_row(
    engine: WhisperEngine,
    audio_path: Path,
    reference: str,
    condition: str,
    sample_id: int,
    snr_db: float | None,
) -> dict[str, Any]:
    """Transcribe one audio file and calculate its WER."""
    from jiwer import wer

    result = engine.transcribe(
        audio_path,
        language="hi",
    )

    score = wer(
        reference,
        result.text,
    )

    return {
        "audio_duration_seconds": result.audio_duration_seconds,
        "condition": condition,
        "hypothesis": result.text,
        "latency_seconds": result.latency_seconds,
        "reference": reference,
        "rtf": result.real_time_factor,
        "sample_id": sample_id,
        "snr_db": snr_db,
        "wer": score,
    }


def _load_references(
    path: Path,
) -> dict[int, dict[str, Any]]:
    """Load cached FLEURS references from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Reference file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    return {
        int(sample_id): value
        for sample_id, value in data.items()
    }


def _save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save benchmark rows to CSV."""
    fieldnames = [
        "audio_duration_seconds",
        "condition",
        "hypothesis",
        "latency_seconds",
        "reference",
        "rtf",
        "sample_id",
        "snr_db",
        "wer",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_summary(
    rows: list[dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    """Calculate and save aggregate benchmark metrics."""
    sample_ids = {
        int(row["sample_id"])
        for row in rows
    }

    summary: dict[str, Any] = {
        "benchmark": {
            "samples": len(sample_ids),
            "conditions": [
                "clean",
                "degraded",
                "enhanced",
            ],
            "snr_db": 10.0,
        },
        "conditions": {},
    }

    for condition in (
        "clean",
        "degraded",
        "enhanced",
    ):
        condition_rows = [
            row
            for row in rows
            if row["condition"] == condition
        ]

        wers = [
            float(row["wer"])
            for row in condition_rows
        ]

        latencies = [
            float(row["latency_seconds"])
            for row in condition_rows
        ]

        rtfs = [
            float(row["rtf"])
            for row in condition_rows
        ]

        summary["conditions"][condition] = {
            "count": len(condition_rows),
            "mean_wer": (
                sum(wers) / len(wers)
            ),
            "mean_latency_seconds": (
                sum(latencies) / len(latencies)
            ),
            "p95_latency_seconds": percentile(
                latencies,
                95,
            ),
            "mean_rtf": (
                sum(rtfs) / len(rtfs)
            ),
        }

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return summary


def _plot_wer(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a mean-WER comparison chart."""
    conditions = [
        "clean",
        "degraded",
        "enhanced",
    ]

    means = []

    for condition in conditions:
        values = [
            float(row["wer"])
            for row in rows
            if row["condition"] == condition
        ]

        means.append(
            sum(values) / len(values)
        )

    plt.figure(figsize=(8, 5))
    plt.bar(
        conditions,
        means,
    )
    plt.ylabel("Mean WER")
    plt.xlabel("Condition")
    plt.title(
        "CallClear Hindi ASR Benchmark — 10 dB"
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )
    plt.close()


def run_local_benchmark(
    results_dir: Path,
    limit: int = 8,
    model_name: str = "base",
) -> dict[str, Any]:
    """Run the benchmark entirely from local cached audio."""
    reference_path = (
        results_dir / "references.json"
    )

    references = _load_references(
        reference_path
    )

    clean_files = sorted(
        results_dir.glob("clean_*.wav")
    )

    if len(clean_files) < limit:
        raise RuntimeError(
            f"Expected at least {limit} clean WAV files, "
            f"found {len(clean_files)}."
        )

    selected_files = clean_files[:limit]

    for clean_path in selected_files:
        sample_id = int(
            clean_path.stem.split("_")[-1]
        )

        if sample_id not in references:
            raise RuntimeError(
                f"No reference found for sample "
                f"{sample_id}."
            )

    engine = WhisperEngine(
        model_name=model_name,
        device="cpu",
        compute_type="int8",
    )

    rows: list[dict[str, Any]] = []

    for clean_path in selected_files:
        sample_id = int(
            clean_path.stem.split("_")[-1]
        )

        reference = str(
            references[sample_id]["reference"]
        ).strip()

        print(
            f"\n{'=' * 60}\n"
            f"Sample {sample_id}\n"
            f"{'=' * 60}"
        )

        clean_row = _transcribe_and_row(
            engine=engine,
            audio_path=clean_path,
            reference=reference,
            condition="clean",
            sample_id=sample_id,
            snr_db=None,
        )

        rows.append(clean_row)

        print(
            "clean    "
            f"WER={clean_row['wer']:.3f} "
            f"latency="
            f"{clean_row['latency_seconds']:.2f}s"
        )

        degraded_path = (
            results_dir
            / f"degraded_{sample_id:04d}_10.wav"
        )

        if not degraded_path.exists():
            samples, sample_rate = sf.read(
                clean_path,
                dtype="float32",
            )

            degraded, rate = degrade_audio(
                samples,
                sample_rate,
                DegradationConfig(
                    snr_db=10.0
                ),
            )

            _write_wav(
                degraded_path,
                degraded,
                rate,
            )

        degraded_row = _transcribe_and_row(
            engine=engine,
            audio_path=degraded_path,
            reference=reference,
            condition="degraded",
            sample_id=sample_id,
            snr_db=10.0,
        )

        rows.append(degraded_row)

        print(
            "degraded "
            f"WER={degraded_row['wer']:.3f} "
            f"latency="
            f"{degraded_row['latency_seconds']:.2f}s"
        )

        enhanced_path = (
            results_dir
            / f"enhanced_{sample_id:04d}_10.wav"
        )

        if not enhanced_path.exists():
            degraded_samples, degraded_rate = sf.read(
                degraded_path,
                dtype="float32",
            )

            enhanced = enhance_audio(
                degraded_samples,
                degraded_rate,
                method="wiener",
            )

            _write_wav(
                enhanced_path,
                enhanced,
                degraded_rate,
            )

        enhanced_row = _transcribe_and_row(
            engine=engine,
            audio_path=enhanced_path,
            reference=reference,
            condition="enhanced",
            sample_id=sample_id,
            snr_db=10.0,
        )

        rows.append(enhanced_row)

        print(
            "enhanced "
            f"WER={enhanced_row['wer']:.3f} "
            f"latency="
            f"{enhanced_row['latency_seconds']:.2f}s"
        )

    csv_path = (
        results_dir / "benchmark_results.csv"
    )

    summary_path = (
        results_dir / "benchmark_summary.json"
    )

    plot_path = (
        results_dir / "wer_vs_snr.png"
    )

    _save_csv(
        rows,
        csv_path,
    )

    summary = _save_summary(
        rows,
        summary_path,
    )

    _plot_wer(
        rows,
        plot_path,
    )

    print(
        "\n"
        + "=" * 60
        + "\nCALLCLEAR LOCAL BENCHMARK COMPLETE\n"
        + "=" * 60
    )

    for condition, metrics in (
        summary["conditions"].items()
    ):
        print(
            f"{condition:>10}: "
            f"mean WER="
            f"{metrics['mean_wer']:.3f}, "
            f"mean latency="
            f"{metrics['mean_latency_seconds']:.2f}s, "
            f"mean RTF="
            f"{metrics['mean_rtf']:.3f}"
        )

    print("=" * 60)

    return summary