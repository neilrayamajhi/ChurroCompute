from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from churro.schema import GroupId, ModelId, Rollout

_PASS_THRESHOLD = 0.5
_WALD_Z_95 = 1.96


@dataclass(frozen=True, slots=True)
class SignalReport:
    model: ModelId | None
    n_tasks: int
    group_size: int
    signal_rate: float
    signal_rate_ci95: float
    dead_too_easy: int
    dead_too_hard: int
    mean_spread: float

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def signal_rate(
    rollouts: Iterable[Rollout], model: ModelId | None = None
) -> SignalReport | None:
    filtered = [r for r in rollouts if model is None or r.model == model]
    groups: dict[GroupId, list[float]] = defaultdict(list)
    for r in filtered:
        groups[r.group_id].append(r.reward)
    n = len(groups)
    if n == 0:
        return None

    live: list[list[float]] = []
    dead_too_easy = 0
    dead_too_hard = 0
    for scores in groups.values():
        if len(set(scores)) > 1:
            live.append(scores)
        elif scores[0] > _PASS_THRESHOLD:
            dead_too_easy += 1
        else:
            dead_too_hard += 1

    rate = len(live) / n
    margin = _WALD_Z_95 * math.sqrt(rate * (1 - rate) / n)
    mean_spread = (
        statistics.mean(statistics.pstdev(s) for s in live) if live else 0.0
    )
    return SignalReport(
        model=model,
        n_tasks=n,
        group_size=statistics.mode(len(s) for s in groups.values()),
        signal_rate=round(rate, 4),
        signal_rate_ci95=round(margin, 4),
        dead_too_easy=dead_too_easy,
        dead_too_hard=dead_too_hard,
        mean_spread=round(mean_spread, 4),
    )
