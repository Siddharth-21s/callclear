"""Run the CallClear local benchmark."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.runner import run_local_benchmark


def main() -> None:
    """Run the local benchmark."""
    results_dir = PROJECT_ROOT / "results"

    run_local_benchmark(
        results_dir=results_dir,
        limit=8,
        model_name="base",
    )


if __name__ == "__main__":
    main()