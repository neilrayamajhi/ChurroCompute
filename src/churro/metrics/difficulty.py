from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from churro.metrics.signal import SignalReport, signal_rate
from churro.schema import EnvId, ModelId, Rollout

_PASS_THRESHOLD = 0.5
_SATURATION_FRACTION = 0.8


@dataclass(frozen=True, slots=True)
class RungReport:
    model: ModelId
    signal: SignalReport | None
    pass_rate: float
    mean_reward: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "signal": self.signal.to_json_dict() if self.signal else None,
            "pass_rate": self.pass_rate,
            "mean_reward": self.mean_reward,
        }


@dataclass(frozen=True, slots=True)
class DifficultyReport:
    env_id: EnvId
    ladder_version: str
    rungs: tuple[RungReport, ...]
    floor: ModelId | None
    ceiling: ModelId | None
    best_signal_model: ModelId | None
    slope: float
    saturated: bool
    possibly_broken: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "env_id": self.env_id,
            "ladder_version": self.ladder_version,
            "rungs": [r.to_json_dict() for r in self.rungs],
            "floor": self.floor,
            "ceiling": self.ceiling,
            "best_signal_model": self.best_signal_model,
            "slope": self.slope,
            "saturated": self.saturated,
            "possibly_broken": self.possibly_broken,
        }


def difficulty_curve(
    rollouts: Iterable[Rollout],
    ladder: Sequence[ModelId],
    ladder_version: str,
) -> DifficultyReport | None:
    rollouts = list(rollouts)
    if not rollouts:
        return None

    env_ids = {r.env_id for r in rollouts}
    if len(env_ids) > 1:
        raise ValueError(f"mixed env_ids in rollouts: {sorted(env_ids)}")
    env_id = next(iter(env_ids))

    rungs = tuple(_build_rung(rollouts, m) for m in ladder)

    floor = _find_floor(rungs)
    ceiling = _find_ceiling(rungs)
    best = _best_signal_model(rungs)
    slope = _compute_slope(rungs, floor, ceiling, ladder)
    saturated = _is_saturated(rungs[-1]) if rungs else False
    possibly_broken = all(
        r.pass_rate == 0.0 and (r.signal is None or r.signal.signal_rate == 0.0)
        for r in rungs
    )

    return DifficultyReport(
        env_id=env_id,
        ladder_version=ladder_version,
        rungs=rungs,
        floor=floor,
        ceiling=ceiling,
        best_signal_model=best,
        slope=round(slope, 4),
        saturated=saturated,
        possibly_broken=possibly_broken,
    )


def _build_rung(rollouts: list[Rollout], model: ModelId) -> RungReport:
    subset = [r for r in rollouts if r.model == model]
    if not subset:
        return RungReport(model=model, signal=None, pass_rate=0.0, mean_reward=0.0)
    passes = sum(1 for r in subset if r.reward > _PASS_THRESHOLD)
    return RungReport(
        model=model,
        signal=signal_rate(subset, model=model),
        pass_rate=round(passes / len(subset), 4),
        mean_reward=round(statistics.mean(r.reward for r in subset), 4),
    )


def _find_floor(rungs: tuple[RungReport, ...]) -> ModelId | None:
    for rung in rungs:
        if rung.signal is not None and rung.pass_rate > 0.0:
            return rung.model
    return None


def _find_ceiling(rungs: tuple[RungReport, ...]) -> ModelId | None:
    ceiling: ModelId | None = None
    for rung in rungs:
        if rung.signal is not None and rung.pass_rate < 1.0:
            ceiling = rung.model
    return ceiling


def _best_signal_model(rungs: tuple[RungReport, ...]) -> ModelId | None:
    live = [r for r in rungs if r.signal is not None and r.signal.signal_rate > 0.0]
    if not live:
        return None
    return max(live, key=lambda r: r.signal.signal_rate).model  # type: ignore[union-attr]


def _compute_slope(
    rungs: tuple[RungReport, ...],
    floor: ModelId | None,
    ceiling: ModelId | None,
    ladder: Sequence[ModelId],
) -> float:
    if floor is None or ceiling is None or floor == ceiling:
        return 0.0
    by_model = {r.model: r for r in rungs}
    floor_pass = by_model[floor].pass_rate
    ceiling_pass = by_model[ceiling].pass_rate
    distance = ladder.index(ceiling) - ladder.index(floor)
    return (ceiling_pass - floor_pass) / distance if distance else 0.0


def _is_saturated(top_rung: RungReport) -> bool:
    signal = top_rung.signal
    if signal is None or signal.n_tasks == 0:
        return False
    return signal.dead_too_easy / signal.n_tasks > _SATURATION_FRACTION
