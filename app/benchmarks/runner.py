"""Local benchmark runner for cached Hindi FLEURS audio."""

import asyncio
import csv
import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from app.asr.engine import WhisperEngine
from app.benchmarks.metrics import percentile
from app.core.config import get_settings
from app.db.mongo import MongoRepository
from app.dsp.enhancement import enhance_audio
from app.dsp.telephony import DegradationConfig, degrade_audio


DEFAULT_SNRS = (20.0, 10.0, 5.0, 0.0)

DEFAULT_METHODS = (
    "none",
    "highpass",
    "spectral_subtraction",
    "wiener",
    "vad",
)


def normalize_transcript(text: str) -> str:
    """Normalize transcript text before normalized-WER calculation.

    Normalization rules:
    1. Apply Unicode NFC normalization.
    2. Lowercase Latin characters.
    3. Remove Unicode punctuation characters.
    4. Treat Hindi danda characters as punctuation.
    5. Preserve Unicode letters, combining marks, numbers, and whitespace.
    6. Collapse repeated whitespace.
    7. Strip leading and trailing whitespace.
    """
    normalized = unicodedata.normalize(
        "NFC",
        str(text),
    )

    normalized = normalized.lower()

    normalized = normalized.replace(
        "।",
        " ",
    ).replace(
        "॥",
        " ",
    )

    normalized = "".join(
        " "
        if unicodedata.category(character).startswith("P")
        else character
        for character in normalized
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


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
    method: str,
    sample_id: int,
    snr_db: float | None,
) -> dict[str, Any]:
    """Transcribe one audio file and calculate raw and normalized WER."""
    from jiwer import wer

    result = engine.transcribe(
        audio_path,
        language="hi",
    )

    raw_score = wer(
        reference,
        result.text,
    )

    normalized_reference = normalize_transcript(
        reference,
    )

    normalized_hypothesis = normalize_transcript(
        result.text,
    )

    normalized_score = wer(
        normalized_reference,
        normalized_hypothesis,
    )

    return {
        "audio_duration_seconds": result.audio_duration_seconds,
        "condition": (
            "clean"
            if snr_db is None
            else "degraded"
            if method == "none"
            else "enhanced"
        ),
        "hypothesis": result.text,
        "latency_seconds": result.latency_seconds,
        "method": method,
        "normalized_hypothesis": normalized_hypothesis,
        "normalized_reference": normalized_reference,
        "reference": reference,
        "rtf": result.real_time_factor,
        "sample_id": sample_id,
        "snr_db": snr_db,
        "wer": raw_score,
        "wer_normalized": normalized_score,
    }


def _load_references(
    path: Path,
) -> dict[int, dict[str, Any]]:
    """Load cached FLEURS references from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            "\n"
            "Benchmark reference data is missing.\n"
            f"Expected file: {path}\n\n"
            "Create it first by running:\n"
            "  uv run python scripts/fetch_references.py\n\n"
            "Then run the benchmark:\n"
            "  uv run python scripts/run_benchmark.py\n"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Benchmark reference file is not valid JSON: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Benchmark reference file must contain a JSON object: {path}"
        )

    return {
        int(sample_id): value
        for sample_id, value in data.items()
    }


def _find_clean_files(
    results_dir: Path,
    limit: int,
) -> list[Path]:
    """Find the cached clean WAV files required by the benchmark."""
    clean_files = sorted(
        results_dir.glob("clean_*.wav")
    )

    if not clean_files:
        raise FileNotFoundError(
            "\n"
            "No cached clean benchmark WAV files were found.\n"
            f"Expected files such as: {results_dir}/clean_0000.wav\n\n"
            "Make sure the clean FLEURS benchmark audio has been "
            "prepared before running the benchmark."
        )

    if len(clean_files) < limit:
        raise RuntimeError(
            "\n"
            f"Benchmark requires {limit} clean WAV files, "
            f"but only {len(clean_files)} were found.\n"
            f"Results directory: {results_dir}\n"
        )

    return clean_files[:limit]


def _validate_reference_coverage(
    clean_files: list[Path],
    references: dict[int, dict[str, Any]],
) -> None:
    """Ensure every selected clean WAV has a reference transcript."""
    missing_ids: list[int] = []

    for clean_path in clean_files:
        sample_id = int(
            clean_path.stem.split("_")[-1]
        )

        if sample_id not in references:
            missing_ids.append(sample_id)

    if missing_ids:
        formatted_ids = ", ".join(
            str(sample_id)
            for sample_id in missing_ids
        )

        raise RuntimeError(
            "\n"
            "Benchmark reference coverage is incomplete.\n"
            f"Missing references for sample IDs: {formatted_ids}\n\n"
            "Regenerate the reference manifest with:\n"
            "  uv run python scripts/fetch_references.py\n"
        )


def _spectral_subtraction(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Apply basic STFT spectral subtraction.

    The noise magnitude estimate is taken from the lowest-energy
    time frames. A spectral floor prevents complete removal of
    frequency bins and reduces musical-noise artifacts.
    """
    x = np.asarray(
        audio,
        dtype=np.float32,
    )

    if x.size == 0:
        return x

    nperseg = min(
        512,
        x.size,
    )

    if nperseg < 32:
        return x.copy()

    noverlap = min(
        nperseg - 1,
        int(nperseg * 0.75),
    )

    frequencies, times, zxx = signal.stft(
        x,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
    )

    del frequencies, times

    magnitude = np.abs(zxx)
    phase = np.angle(zxx)

    frame_energy = np.mean(
        magnitude**2,
        axis=0,
    )

    noise_frame_count = max(
        1,
        int(np.ceil(magnitude.shape[1] * 0.20)),
    )

    noise_indices = np.argsort(
        frame_energy
    )[:noise_frame_count]

    noise_magnitude = np.median(
        magnitude[:, noise_indices],
        axis=1,
        keepdims=True,
    )

    estimated_signal = (
        magnitude - noise_magnitude
    )

    spectral_floor = (
        0.10 * magnitude
    )

    estimated_signal = np.maximum(
        estimated_signal,
        spectral_floor,
    )

    reconstructed = estimated_signal * np.exp(
        1j * phase
    )

    _, output = signal.istft(
        reconstructed,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )

    output = output[: x.size]

    if output.size < x.size:
        output = np.pad(
            output,
            (0, x.size - output.size),
        )

    return np.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _apply_enhancement(
    audio: np.ndarray,
    sample_rate: int,
    method: str,
) -> np.ndarray:
    """Apply the requested benchmark enhancement method."""
    if method == "none":
        return np.asarray(
            audio,
            dtype=np.float32,
        )

    if method == "spectral_subtraction":
        enhanced = _spectral_subtraction(
            audio,
            sample_rate,
        )
    else:
        enhanced = enhance_audio(
            audio,
            sample_rate,
            method=method,
        )

    return np.nan_to_num(
        enhanced,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save benchmark rows in long format."""
    fieldnames = [
        "audio_duration_seconds",
        "condition",
        "hypothesis",
        "latency_seconds",
        "method",
        "normalized_hypothesis",
        "normalized_reference",
        "reference",
        "rtf",
        "sample_id",
        "snr_db",
        "wer",
        "wer_normalized",
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


def _mean(
    values: list[float],
) -> float:
    """Calculate an arithmetic mean."""
    if not values:
        return 0.0

    return sum(values) / len(values)


def _save_summary(
    rows: list[dict[str, Any]],
    path: Path,
    snrs: tuple[float, ...],
    methods: tuple[str, ...],
    requested_limit: int,
) -> dict[str, Any]:
    """Calculate and save aggregate benchmark metrics."""
    sample_ids = {
        int(row["sample_id"])
        for row in rows
    }

    summary: dict[str, Any] = {
        "benchmark": {
            "samples": len(sample_ids),
            "requested_samples": requested_limit,
            "snrs_db": list(snrs),
            "methods": list(methods),
            "rows": len(rows),
            "metrics": [
                "raw_wer",
                "normalized_wer",
                "latency",
                "rtf",
            ],
            "normalization": {
                "unicode": "NFC",
                "lowercase_latin": True,
                "remove_punctuation": True,
                "remove_hindi_danda": True,
                "collapse_whitespace": True,
            },
        },
        "conditions": {},
    }

    for snr_db in snrs:
        snr_key = str(
            snr_db
        ).rstrip("0").rstrip(".")

        for method in methods:
            condition_rows = [
                row
                for row in rows
                if row["snr_db"] == snr_db
                and row["method"] == method
            ]

            if not condition_rows:
                continue

            raw_wers = [
                float(row["wer"])
                for row in condition_rows
            ]

            normalized_wers = [
                float(row["wer_normalized"])
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

            key = f"{snr_key}dB/{method}"

            summary["conditions"][key] = {
                "snr_db": snr_db,
                "method": method,
                "count": len(condition_rows),
                "mean_wer": _mean(raw_wers),
                "mean_wer_normalized": _mean(
                    normalized_wers
                ),
                "mean_latency_seconds": _mean(
                    latencies
                ),
                "p50_latency_seconds": percentile(
                    latencies,
                    50,
                ),
                "p95_latency_seconds": percentile(
                    latencies,
                    95,
                ),
                "mean_rtf": _mean(rtfs),
            }

    clean_rows = [
        row
        for row in rows
        if row["condition"] == "clean"
    ]

    if clean_rows:
        clean_raw_wers = [
            float(row["wer"])
            for row in clean_rows
        ]

        clean_normalized_wers = [
            float(row["wer_normalized"])
            for row in clean_rows
        ]

        clean_latencies = [
            float(row["latency_seconds"])
            for row in clean_rows
        ]

        clean_rtfs = [
            float(row["rtf"])
            for row in clean_rows
        ]

        summary["clean_baseline"] = {
            "count": len(clean_rows),
            "mean_wer": _mean(
                clean_raw_wers
            ),
            "mean_wer_normalized": _mean(
                clean_normalized_wers
            ),
            "mean_latency_seconds": _mean(
                clean_latencies
            ),
            "p50_latency_seconds": percentile(
                clean_latencies,
                50,
            ),
            "p95_latency_seconds": percentile(
                clean_latencies,
                95,
            ),
            "mean_rtf": _mean(
                clean_rtfs
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
    snrs: tuple[float, ...],
    methods: tuple[str, ...],
) -> None:
    """Generate raw WER-versus-SNR lines for every method."""
    plt.figure(
        figsize=(9, 6)
    )

    plotted = False

    for method in methods:
        points: list[tuple[float, float]] = []

        for snr_db in snrs:
            values = [
                float(row["wer"])
                for row in rows
                if row["method"] == method
                and row["snr_db"] == snr_db
            ]

            if values:
                points.append(
                    (
                        snr_db,
                        _mean(values),
                    )
                )

        if not points:
            continue

        plotted = True

        points.sort(
            key=lambda item: item[0]
        )

        x_values = [
            point[0]
            for point in points
        ]

        y_values = [
            point[1]
            for point in points
        ]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            label=method,
        )

    if not plotted:
        raise RuntimeError(
            "No benchmark rows were available for WER-vs-SNR plotting."
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean raw WER")
    plt.title(
        "CallClear Hindi ASR Benchmark — Raw WER vs SNR"
    )
    plt.gca().invert_xaxis()
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )
    plt.close()


async def _persist_benchmark_to_mongo(
    run_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    model_name: str,
    requested_limit: int,
    snrs: tuple[float, ...],
    methods: tuple[str, ...],
    seed: int,
) -> None:
    """Persist one completed benchmark run and its rows to MongoDB."""
    settings = get_settings()

    repository = MongoRepository(
        settings.mongodb_uri,
        settings.mongodb_database,
    )

    created_at = datetime.now(timezone.utc)

    try:
        await repository.connect()

        await repository.insert_run(
            {
                "run_id": run_id,
                "created_at": created_at,
                "benchmark": "local",
                "model": model_name,
                "requested_limit": requested_limit,
                "snrs_db": list(snrs),
                "methods": list(methods),
                "seed": seed,
                "sample_count": summary["benchmark"]["samples"],
                "row_count": summary["benchmark"]["rows"],
                "status": "completed",
            }
        )

        for row in rows:
            document = dict(row)

            document.update(
                {
                    "run_id": run_id,
                    "created_at": created_at,
                    "model": model_name,
                }
            )

            await repository.insert_benchmark_result(
                document
            )

    finally:
        await repository.close()


def _persist_benchmark_if_available(
    run_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    model_name: str,
    requested_limit: int,
    snrs: tuple[float, ...],
    methods: tuple[str, ...],
    seed: int,
) -> bool:
    """Persist benchmark data when MongoDB is available.

    MongoDB failures are intentionally non-fatal because the benchmark
    always has CSV/JSON/PNG artifacts as its local source of truth.
    """
    try:
        asyncio.run(
            _persist_benchmark_to_mongo(
                run_id=run_id,
                rows=rows,
                summary=summary,
                model_name=model_name,
                requested_limit=requested_limit,
                snrs=snrs,
                methods=methods,
                seed=seed,
            )
        )
    except Exception as exc:
        print(
            "\nMongoDB unavailable; benchmark results were saved "
            "locally only."
        )
        print(
            f"MongoDB detail: {exc}"
        )
        return False

    print(
        f"\nMongoDB persistence complete: "
        f"run_id={run_id}"
    )

    return True


def run_local_benchmark(
    results_dir: Path,
    limit: int = 8,
    model_name: str = "base",
    snrs: tuple[float, ...] = DEFAULT_SNRS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the reproducible local benchmark sweep."""
    if limit < 1:
        raise ValueError(
            "Benchmark limit must be at least 1."
        )

    if not snrs:
        raise ValueError(
            "At least one SNR value is required."
        )

    if not methods:
        raise ValueError(
            "At least one benchmark method is required."
        )

    unsupported_methods = set(methods) - set(
        DEFAULT_METHODS
    )

    if unsupported_methods:
        names = ", ".join(
            sorted(unsupported_methods)
        )

        raise ValueError(
            "Unsupported benchmark methods: "
            f"{names}. Supported methods: "
            f"{', '.join(DEFAULT_METHODS)}"
        )

    run_id = uuid.uuid4().hex

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_path = (
        results_dir / "references.json"
    )

    references = _load_references(
        reference_path
    )

    selected_files = _find_clean_files(
        results_dir=results_dir,
        limit=limit,
    )

    _validate_reference_coverage(
        clean_files=selected_files,
        references=references,
    )

    engine = WhisperEngine(
        model_name=model_name,
        device="cpu",
        compute_type="int8",
    )

    rows: list[dict[str, Any]] = []

    print(
        "\n"
        + "=" * 72
        + "\nCALLCLEAR BENCHMARK SWEEP\n"
        + "=" * 72
    )

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Samples: {len(selected_files)}"
    )

    print(
        f"SNRs: {', '.join(str(value) for value in snrs)} dB"
    )

    print(
        f"Methods: {', '.join(methods)}"
    )

    print(
        f"Seed: {seed}"
    )

    print(
        "Normalization: NFC + lowercase Latin + "
        "punctuation/danda removal + whitespace collapse"
    )

    print(
        "Total non-clean evaluations: "
        f"{len(selected_files) * len(snrs) * len(methods)}"
    )

    for clean_path in selected_files:
        sample_id = int(
            clean_path.stem.split("_")[-1]
        )

        reference = str(
            references[sample_id]["reference"]
        ).strip()

        print(
            f"\n{'=' * 72}\n"
            f"Sample {sample_id}\n"
            f"{'=' * 72}"
        )

        clean_row = _transcribe_and_row(
            engine=engine,
            audio_path=clean_path,
            reference=reference,
            method="clean",
            sample_id=sample_id,
            snr_db=None,
        )

        rows.append(clean_row)

        print(
            "clean    "
            f"raw-WER={clean_row['wer']:.3f} "
            f"normalized-WER="
            f"{clean_row['wer_normalized']:.3f} "
            f"latency="
            f"{clean_row['latency_seconds']:.2f}s "
            f"RTF={clean_row['rtf']:.3f}"
        )

        clean_samples, clean_rate = sf.read(
            clean_path,
            dtype="float32",
        )

        for snr_index, snr_db in enumerate(snrs):
            degradation_seed = (
                seed
                + sample_id * 100_000
                + snr_index * 1_000
            )

            degraded_path = (
                results_dir
                / (
                    f"degraded_{sample_id:04d}_"
                    f"{snr_db:g}.wav"
                )
            )

            if not degraded_path.exists():
                degraded, degraded_rate = degrade_audio(
                    clean_samples,
                    clean_rate,
                    DegradationConfig(
                        snr_db=snr_db,
                        seed=degradation_seed,
                    ),
                )

                _write_wav(
                    degraded_path,
                    degraded,
                    degraded_rate,
                )

            for method in methods:
                if method == "none":
                    audio_path = degraded_path
                else:
                    enhanced_path = (
                        results_dir
                        / (
                            f"enhanced_{sample_id:04d}_"
                            f"{snr_db:g}_{method}.wav"
                        )
                    )

                    if not enhanced_path.exists():
                        degraded_samples, degraded_rate = sf.read(
                            degraded_path,
                            dtype="float32",
                        )

                        enhanced = _apply_enhancement(
                            degraded_samples,
                            degraded_rate,
                            method,
                        )

                        _write_wav(
                            enhanced_path,
                            enhanced,
                            degraded_rate,
                        )

                    audio_path = enhanced_path

                row = _transcribe_and_row(
                    engine=engine,
                    audio_path=audio_path,
                    reference=reference,
                    method=method,
                    sample_id=sample_id,
                    snr_db=snr_db,
                )

                rows.append(row)

                print(
                    f"{snr_db:>5g} dB "
                    f"{method:<21} "
                    f"raw-WER={row['wer']:.3f} "
                    f"normalized-WER="
                    f"{row['wer_normalized']:.3f} "
                    f"latency="
                    f"{row['latency_seconds']:.2f}s "
                    f"RTF={row['rtf']:.3f}"
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
        rows=rows,
        path=summary_path,
        snrs=snrs,
        methods=methods,
        requested_limit=limit,
    )

    _plot_wer(
        rows=rows,
        output_path=plot_path,
        snrs=snrs,
        methods=methods,
    )

    _persist_benchmark_if_available(
        run_id=run_id,
        rows=rows,
        summary=summary,
        model_name=model_name,
        requested_limit=limit,
        snrs=snrs,
        methods=methods,
        seed=seed,
    )

    print(
        "\n"
        + "=" * 72
        + "\nCALLCLEAR LOCAL BENCHMARK COMPLETE\n"
        + "=" * 72
    )

    if "clean_baseline" in summary:
        baseline = summary["clean_baseline"]

        print(
            "Clean baseline: "
            f"raw-WER={baseline['mean_wer']:.3f}, "
            f"normalized-WER="
            f"{baseline['mean_wer_normalized']:.3f}, "
            f"p50 latency="
            f"{baseline['p50_latency_seconds']:.2f}s, "
            f"p95 latency="
            f"{baseline['p95_latency_seconds']:.2f}s, "
            f"mean RTF="
            f"{baseline['mean_rtf']:.3f}"
        )

    print(
        "\nGenerated artifacts:"
    )

    print(
        f"  CSV:     {csv_path}"
    )

    print(
        f"  Summary: {summary_path}"
    )

    print(
        f"  Plot:    {plot_path}"
    )

    print("=" * 72)

    return summary