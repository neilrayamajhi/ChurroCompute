from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    n_items_tested: int
    n_repeats: int
    disagreement_rate: float
    max_spread: float
    deterministic: bool


def consistency(
    score_fn: Callable[[T], float], items: Sequence[T], n_repeats: int = 5
) -> ConsistencyReport:
    spreads = []
    for item in items:
        scores = [score_fn(item) for _ in range(n_repeats)]
        spreads.append(max(scores) - min(scores))
    disagreeing = sum(1 for s in spreads if s > 0)
    return ConsistencyReport(
        n_items_tested=len(items),
        n_repeats=n_repeats,
        disagreement_rate=round(disagreeing / len(items), 4) if items else 0.0,
        max_spread=round(max(spreads, default=0.0), 4),
        deterministic=disagreeing == 0,
    )
