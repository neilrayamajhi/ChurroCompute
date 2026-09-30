from __future__ import annotations

from churro.leaderboard import LeaderboardRow, summarize

MIN_TASKS = 10


def _rung(
    model: str, tasks: int, signal: float, ci: float, pass_rate: float, easy: int = 0
) -> dict[str, object]:
    return {
        "model": model,
        "pass_rate": pass_rate,
        "signal": {
            "n_tasks": tasks,
            "signal_rate": signal,
            "signal_rate_ci95": ci,
            "dead_too_easy": easy,
        },
    }


def _fingerprint(rungs: list[dict[str, object]], **extra: object) -> dict[str, object]:
    return {
        "env_id": "toy-env",
        "grader_check": "ok",
        "difficulty": {"rungs": rungs, "saturated": False},
        "shape": {"overall": {"effectively_binary": False}},
        **extra,
    }


class TestSummarize:
    def test_picks_the_highest_signal_rung_with_enough_tasks(self) -> None:
        fp = _fingerprint(
            [
                _rung("qwen3:0.6b", 30, 0.40, 0.18, 0.2),
                _rung("qwen3:4b", 3, 1.00, 0.00, 0.7),
                _rung("qwen3:8b", 30, 0.60, 0.17, 0.5),
            ]
        )

        assert summarize(fp, min_tasks=MIN_TASKS) == LeaderboardRow(
            env_id="toy-env",
            grader_check="ok",
            best_model="qwen3:8b",
            best_signal=0.60,
            best_signal_ci95=0.17,
            pass_range=(0.2, 0.5),
            rungs_measured=2,
            saturated=False,
            binary_reward=False,
        )

    def test_rungs_without_signal_are_ignored(self) -> None:
        fp = _fingerprint(
            [
                _rung("qwen3:0.6b", 30, 0.5, 0.2, 0.4),
                {"model": "qwen3:14b", "pass_rate": 0.0, "signal": None},
            ]
        )

        row = summarize(fp, min_tasks=MIN_TASKS)

        assert (row.best_model, row.rungs_measured, row.pass_range) == ("qwen3:0.6b", 1, (0.4, 0.4))

    def test_no_rung_with_enough_tasks_leaves_best_model_empty(self) -> None:
        fp = _fingerprint([_rung("qwen3:0.6b", 1, 1.0, 0.0, 1.0)])

        row = summarize(fp, min_tasks=MIN_TASKS)

        assert (row.best_model, row.best_signal, row.rungs_measured) == (None, None, 0)

    def test_saturated_when_top_measured_rung_is_mostly_too_easy(self) -> None:
        fp = _fingerprint(
            [_rung("qwen3:0.6b", 30, 0.3, 0.1, 0.5), _rung("qwen3:14b", 30, 0.1, 0.1, 1.0, easy=27)]
        )

        assert summarize(fp, min_tasks=MIN_TASKS).saturated is True

    def test_saturation_ignores_a_top_rung_with_too_few_tasks(self) -> None:
        fp = _fingerprint(
            [_rung("qwen3:8b", 30, 0.7, 0.16, 0.48, easy=3), _rung("qwen3:14b", 1, 0.0, 0.0, 1.0, easy=1)]
        )
        fp["difficulty"]["saturated"] = True  # type: ignore[index]

        assert summarize(fp, min_tasks=MIN_TASKS).saturated is False

    def test_carries_the_binary_reward_flag(self) -> None:
        fp = _fingerprint([_rung("qwen3:0.6b", 30, 0.3, 0.1, 0.9)])
        fp["shape"]["overall"]["effectively_binary"] = True  # type: ignore[index]

        assert summarize(fp, min_tasks=MIN_TASKS).binary_reward is True
