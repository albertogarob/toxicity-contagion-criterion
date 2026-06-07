"""Small statistical helpers shared across the analyses.

Centralizes the Pearson correlation and the permutation-null summary so the criterion and
validation modules do not each re-implement them. The permutation tests themselves live in the
analysis modules (the draw pattern is analysis-specific), but they report results through
:func:`null_summary`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient; returns 0.0 for a degenerate (zero-variance) input."""
    n = len(x)
    if n == 0:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0


def null_summary(obs: float, null: Sequence[float]) -> dict[str, float]:
    """Summarize an observed statistic against a permutation null distribution.

    Returns ``{"mean", "sd", "z", "p"}`` where ``p`` is the one-sided empirical p-value
    ``(#{null >= obs} + 1) / (B + 1)``.
    """
    n = len(null)
    m = sum(null) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in null) / n)
    z = (obs - m) / sd if sd else float("inf")
    p = (sum(1 for v in null if v >= obs) + 1) / (n + 1)
    return {"mean": m, "sd": sd, "z": z, "p": p}
