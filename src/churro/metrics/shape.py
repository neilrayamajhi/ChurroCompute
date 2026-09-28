from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from churro.schema import ModelId, Rollout

_BINARY_BINS = 2.5


@dataclass(frozen=True, slots=True)
class ShapeReport:
    model: ModelId | None
    n_rollouts: int
    distinct_values: int
    effective_bins: float
    effectively_binary: bool
    frac_at_extremes: float
    histogram: dict[float, int]


def reward_shape(
    rollouts: Iterable[Rollout], model: ModelId | None = None
) -> ShapeReport | None:
    scores = [round(r.reward, 4) for r in rollouts if model is None or r.model == model]
    n = len(scores)
    if n == 0:
        return None
    counts = Counter(scores)
    # exp(entropy) weights values by frequency, so 11 one-off scores around a
    # dominant value don't make a binary reward look graded.
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    effective_bins = math.exp(entropy)
    return ShapeReport(
        model=model,
        n_rollouts=n,
        distinct_values=len(counts),
        effective_bins=round(effective_bins, 2),
        effectively_binary=effective_bins < _BINARY_BINS,
        frac_at_extremes=round((counts.get(0.0, 0) + counts.get(1.0, 0)) / n, 4),
        histogram=dict(sorted(counts.items())),
    )
