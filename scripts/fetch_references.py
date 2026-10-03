"""Recover references for the cached FLEURS benchmark samples."""

import json
from pathlib import Path

from datasets import Audio, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REFERENCE_PATH = RESULTS_DIR / "references.json"
SAMPLE_COUNT = 8


def main() -> None:
    """Fetch references for the cached benchmark WAV files."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_files = sorted(
        RESULTS_DIR.glob("clean_*.wav")
    )

    if len(clean_files) < SAMPLE_COUNT:
        raise RuntimeError(
            "\n"
            f"Expected at least {SAMPLE_COUNT} cached clean WAV files, "
            f"found {len(clean_files)}.\n"
            f"Expected directory: {RESULTS_DIR}\n\n"
            "Prepare the clean benchmark WAV files first."
        )

    print(
        f"Recovering references for the first "
        f"{SAMPLE_COUNT} FLEURS samples..."
    )

    dataset = load_dataset(
        "google/fleurs",
        "hi_in",
        split="test",
        streaming=True,
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False),
    )

    references: dict[str, dict[str, object]] = {}

    for sample_index, item in enumerate(dataset):
        if sample_index >= SAMPLE_COUNT:
            break

        references[str(sample_index)] = {
            "fleurs_id": int(item["id"]),
            "reference": str(
                item["transcription"]
            ).strip(),
        }

        print(
            f"\nSample {sample_index}"
            f"\n  FLEURS ID: {item['id']}"
            f"\n  Reference: "
            f"{item['transcription']}"
        )

    if len(references) != SAMPLE_COUNT:
        raise RuntimeError(
            "\n"
            f"Only recovered {len(references)} references; "
            f"expected {SAMPLE_COUNT}."
        )

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

    print(
        f"\nSaved references to {REFERENCE_PATH}"
    )
    print(
        "The benchmark can now be run with:"
    )
    print(
        "  uv run python scripts/run_benchmark.py"
    )


if __name__ == "__main__":
    main()