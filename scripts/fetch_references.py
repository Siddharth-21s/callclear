"""Recover references for the eight cached FLEURS samples."""

import json
from pathlib import Path

from datasets import Audio, load_dataset


RESULTS_DIR = Path("results")
REFERENCE_PATH = RESULTS_DIR / "references.json"
SAMPLE_COUNT = 8


def main() -> None:
    """Fetch the first cached FLEURS references in dataset order."""
    clean_files = sorted(RESULTS_DIR.glob("clean_*.wav"))

    if len(clean_files) < SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected at least {SAMPLE_COUNT} cached clean WAV files, "
            f"found {len(clean_files)}."
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

    references = {}

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
            f"  FLEURS ID: {item['id']}"
            f"\n  Reference: "
            f"{item['transcription']}"
        )

    if len(references) != SAMPLE_COUNT:
        raise RuntimeError(
            f"Only recovered {len(references)} references."
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


if __name__ == "__main__":
    main()