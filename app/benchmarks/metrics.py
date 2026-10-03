"""Benchmark metrics and summary helpers."""

import math
from statistics import median


def percentile(values: list[float], p: float) -> float:
    """Compute a linear-interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latency(values: list[float]) -> dict[str, float]:
    """Return p50 and p95 latency."""
    return {"p50": percentile(values, 50), "p95": percentile(values, 95)}


def median_rtf(values: list[float]) -> float:
    """Return median real-time factor."""
    return float(median(values)) if values else 0.0
