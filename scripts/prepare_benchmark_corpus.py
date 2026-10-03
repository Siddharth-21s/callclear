"""Prepare a reproducible local Hindi FLEURS benchmark corpus."""

from __future__ import annotations

import io
import json
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REFERENCE_PATH = RESULTS_DIR / "references.json"

TARGET_COUNT = 200
DATASET_NAME = "google/fleurs"
DATASET_CONFIG = "hi_in"
DATASET_SPLIT = "test"


def _load_existing_references() -> dict[str, dict[str, object]]:
    """Load the existing local reference mapping when available."""
    if not REFERENCE_PATH.exists():
        return {}

    with REFERENCE_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    return {
        str(key): dict(value)
        for key, value in data.items()
    }


def _validate_existing_files(
    references: dict[str, dict[str, object]],
) -> None:
    """Validate the existing clean WAV corpus."""
    for sample_id in sorted(references, key=int):
        path = RESULTS_DIR / f"clean_{int(sample_id):04d}.wav"

        if not path.exists():
            raise RuntimeError(
                f"Reference {sample_id} exists but WAV is missing: {path}"
            )

        info = sf.info(path)

        if info.samplerate != 16_000:
            raise RuntimeError(
                f"{path} has sample rate {info.samplerate}; "
                "expected 16000 Hz."
            )

        if info.channels != 1:
            raise RuntimeError(
                f"{path} has {info.channels} channels; "
                "expected mono audio."
            )

        if info.subtype != "PCM_16":
            raise RuntimeError(
                f"{path} has subtype {info.subtype}; "
                "expected PCM_16."
            )


def _save_references(
    references: dict[str, dict[str, object]],
) -> None:
    """Save the complete benchmark reference mapping."""
    with REFERENCE_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            references,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    """Stream FLEURS and prepare the local benchmark corpus."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    references = _load_existing_references()

    _validate_existing_files(references)

    existing_count = len(references)

    if existing_count >= TARGET_COUNT:
        print(
            f"Corpus already contains {existing_count} references."
        )
        print(
            f"Target count: {TARGET_COUNT}"
        )
        return

    print("=" * 72)
    print("CALLCLEAR BENCHMARK CORPUS PREPARATION")
    print("=" * 72)
    print(f"Dataset:        {DATASET_NAME}")
    print(f"Configuration:  {DATASET_CONFIG}")
    print(f"Split:          {DATASET_SPLIT}")
    print(f"Existing:       {existing_count}")
    print(f"Target:         {TARGET_COUNT}")
    print(
        f"Additional:     {TARGET_COUNT - existing_count}"
    )
    print()
    print("Streaming FLEURS dataset...")
    print()

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False),
    )

    for _dataset_index, item in enumerate(dataset):
        if len(references) >= TARGET_COUNT:
            break

        sample_id = len(references)
        sample_key = str(sample_id)

        if sample_key in references:
            continue

        audio_info = item["audio"]

        raw_path = audio_info["path"]

        if raw_path is None:
            raise RuntimeError(
                f"FLEURS item {item['id']} has no audio path."
            )

        audio_bytes = audio_info["bytes"]

        if audio_bytes is None:
            raise RuntimeError(
                f"FLEURS item {item['id']} did not provide "
                "streamed audio bytes."
            )

        audio, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )

        if audio.ndim != 1:
            raise RuntimeError(
                f"FLEURS item {item['id']} is not mono."
            )

        if sample_rate != 16_000:
            raise RuntimeError(
                f"FLEURS item {item['id']} has sample rate "
                f"{sample_rate}; expected 16000 Hz."
            )

        output_path = (
            RESULTS_DIR / f"clean_{sample_id:04d}.wav"
        )

        if output_path.exists():
            info = sf.info(output_path)

            if (
                info.samplerate != 16_000
                or info.channels != 1
                or info.subtype != "PCM_16"
            ):
                raise RuntimeError(
                    f"Existing file has invalid format: {output_path}"
                )
        else:
            sf.write(
                output_path,
                audio,
                sample_rate,
                subtype="PCM_16",
            )

        references[sample_key] = {
            "fleurs_id": int(item["id"]),
            "reference": str(
                item["transcription"]
            ).strip(),
        }

        print(
            f"[{len(references):>3}/{TARGET_COUNT}] "
            f"sample={sample_id:04d} "
            f"FLEURS_ID={item['id']} "
            f"duration={len(audio) / sample_rate:.2f}s"
        )

        # Save periodically so an interrupted run does not lose
        # all progress.
        if len(references) % 10 == 0:
            _save_references(references)

    if len(references) != TARGET_COUNT:
        raise RuntimeError(
            f"Only prepared {len(references)} samples; "
            f"expected {TARGET_COUNT}."
        )

    _save_references(references)

    _validate_existing_files(references)

    print()
    print("=" * 72)
    print("CORPUS PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Samples:     {len(references)}")
    print(f"References:  {REFERENCE_PATH}")
    print(f"Audio:       {RESULTS_DIR / 'clean_0000.wav'} ...")
    print(
        f"Audio count: "
        f"{len(list(RESULTS_DIR.glob('clean_*.wav')))}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()