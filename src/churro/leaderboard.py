from __future__ import annotations

from dataclasses import dataclass


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
        saturated=fingerprint["difficulty"]["saturated"],
        binary_reward=fingerprint["shape"]["overall"]["effectively_binary"],
    )
