from __future__ import annotations

from dataclasses import dataclass

# Same rule as difficulty._is_saturated: most of the top rung's tasks are
# solved on every attempt.
_SATURATION_FRACTION = 0.8


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    env_id: str
    grader_check: str | None
    best_model: str | None
    best_signal: float | None
    best_signal_ci95: float | None
    pass_range: tuple[float, float] | None
    rungs_measured: int
    saturated: bool
    binary_reward: bool


def summarize(fingerprint: dict, min_tasks: int = 10) -> LeaderboardRow:
    # Rungs cut short by the per-rung time cap can have 1-4 tasks; one lucky
    # 100% +/- 0% there would otherwise be crowned the best model to train.
    measured = [
        r
        for r in fingerprint["difficulty"]["rungs"]
        if r["signal"] is not None and r["signal"]["n_tasks"] >= min_tasks
    ]
    best = max(measured, key=lambda r: r["signal"]["signal_rate"], default=None)
    passes = [r["pass_rate"] for r in measured]
    return LeaderboardRow(
        env_id=fingerprint["env_id"],
        grader_check=fingerprint.get("grader_check"),
        best_model=best["model"] if best else None,
        best_signal=best["signal"]["signal_rate"] if best else None,
        best_signal_ci95=best["signal"]["signal_rate_ci95"] if best else None,
        pass_range=(min(passes), max(passes)) if passes else None,
        rungs_measured=len(measured),
        saturated=_top_rung_saturated(measured),
        binary_reward=fingerprint["shape"]["overall"]["effectively_binary"],
    )


def _top_rung_saturated(measured: list[dict]) -> bool:
    # Judged on the largest rung with enough tasks, not whichever rung is
    # last: a 14b rung cut off after one easy task isn't saturation.
    if not measured:
        return False
    top = measured[-1]["signal"]
    return top["dead_too_easy"] / top["n_tasks"] > _SATURATION_FRACTION
