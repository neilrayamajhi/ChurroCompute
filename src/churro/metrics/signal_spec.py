from __future__ import annotations

import math

import pytest

from churro.metrics.signal import SignalReport, signal_rate
from churro.schema import (
    CHURRO_VERSION,
    EnvId,
    GroupId,
    ModelId,
    Rollout,
    TaskId,
    make_group_id,
)


def _rollout(*, task: str, reward: float, model: str = "qwen3:1.7b") -> Rollout:
    env = EnvId("gsm8k")
    task_id = TaskId(task)
    model_id = ModelId(model)
    return Rollout(
        env_id=env,
        task_id=task_id,
        group_id=make_group_id(env, task_id, model_id),
        rollout_idx=0,
        model=model_id,
        reward=reward,
        num_turns=1,
        completion_tokens=100,
        collected_at="2026-08-28T00:00:00+00:00",
        churro_version=CHURRO_VERSION,
    )


def _group(task: str, scores: list[float], model: str = "qwen3:1.7b") -> list[Rollout]:
    return [_rollout(task=task, reward=s, model=model) for s in scores]


class TestSignalRate:
    def test_returns_none_when_no_rollouts(self) -> None:
        assert signal_rate([]) is None

    def test_all_pass_group_is_dead_too_easy(self) -> None:
        rollouts = _group("t1", [1.0, 1.0, 1.0, 1.0])
        report = signal_rate(rollouts)
        assert report == SignalReport(
            model=None,
            n_tasks=1,
            group_size=4,
            signal_rate=0.0,
            signal_rate_ci95=0.0,
            dead_too_easy=1,
            dead_too_hard=0,
            mean_spread=0.0,
        )

    def test_all_fail_group_is_dead_too_hard(self) -> None:
        rollouts = _group("t1", [0.0, 0.0, 0.0, 0.0])
        report = signal_rate(rollouts)
        assert report is not None
        assert (report.dead_too_easy, report.dead_too_hard) == (0, 1)
        assert report.signal_rate == 0.0

    def test_disagreeing_group_is_live(self) -> None:
        rollouts = _group("t1", [1.0, 0.0, 1.0, 0.0])
        report = signal_rate(rollouts)
        assert report is not None
        assert report.signal_rate == 1.0
        assert (report.dead_too_easy, report.dead_too_hard) == (0, 0)

    def test_signal_rate_is_live_over_total_groups(self) -> None:
        rollouts = [
            *_group("live1", [1.0, 0.0]),
            *_group("live2", [1.0, 0.0]),
            *_group("easy1", [1.0, 1.0]),
            *_group("hard1", [0.0, 0.0]),
        ]
        report = signal_rate(rollouts)
        assert report is not None
        assert report.n_tasks == 4
        assert report.signal_rate == 0.5
        assert report.dead_too_easy == 1
        assert report.dead_too_hard == 1

    def test_mean_spread_is_mean_of_pstdev_over_live_groups(self) -> None:
        rollouts = [
            *_group("live1", [1.0, 0.0, 1.0, 0.0]),  # pstdev = 0.5
            *_group("live2", [1.0, 1.0, 1.0, 0.0]),  # pstdev = sqrt(3/16)
            *_group("easy", [1.0, 1.0, 1.0, 1.0]),  # excluded from mean_spread
        ]
        report = signal_rate(rollouts)
        expected = round((0.5 + math.sqrt(3 / 16)) / 2, 4)
        assert report is not None
        assert report.mean_spread == expected

    def test_group_size_is_mode_of_per_group_sizes(self) -> None:
        rollouts = [
            *_group("t1", [1.0, 0.0, 1.0, 0.0]),  # 4
            *_group("t2", [1.0, 0.0, 1.0, 0.0]),  # 4
            *_group("t3", [1.0, 0.0]),  # 2 — outlier
        ]
        report = signal_rate(rollouts)
        assert report is not None
        assert report.group_size == 4

    def test_wald_ci_matches_hand_computation(self) -> None:
        rollouts = [
            *_group("live1", [1.0, 0.0]),
            *_group("live2", [1.0, 0.0]),
            *_group("easy1", [1.0, 1.0]),
            *_group("hard1", [0.0, 0.0]),
        ]
        report = signal_rate(rollouts)
        p, n = 0.5, 4
        expected_margin = round(1.96 * math.sqrt(p * (1 - p) / n), 4)
        assert report is not None
        assert report.signal_rate_ci95 == expected_margin

    def test_model_filter_subsets_before_grouping(self) -> None:
        rollouts = [
            *_group("t1", [1.0, 0.0], model="qwen3:1.7b"),
            *_group("t1", [1.0, 1.0], model="qwen3:8b"),
        ]
        weak = signal_rate(rollouts, model=ModelId("qwen3:1.7b"))
        strong = signal_rate(rollouts, model=ModelId("qwen3:8b"))
        assert weak is not None and strong is not None
        assert weak.n_tasks == 1
        assert weak.signal_rate == 1.0
        assert strong.n_tasks == 1
        assert strong.signal_rate == 0.0
        assert strong.dead_too_easy == 1

    def test_threshold_boundary_below_half_is_too_hard(self) -> None:
        rollouts = _group("t1", [0.5, 0.5, 0.5, 0.5])
        report = signal_rate(rollouts)
        assert report is not None
        assert (report.dead_too_easy, report.dead_too_hard) == (0, 1)

    def test_threshold_boundary_above_half_is_too_easy(self) -> None:
        rollouts = _group("t1", [0.51, 0.51, 0.51, 0.51])
        report = signal_rate(rollouts)
        assert report is not None
        assert (report.dead_too_easy, report.dead_too_hard) == (1, 0)

    def test_model_field_carries_through_to_report(self) -> None:
        rollouts = _group("t1", [1.0, 0.0], model="qwen3:1.7b")
        report = signal_rate(rollouts, model=ModelId("qwen3:1.7b"))
        assert report is not None
        assert report.model == "qwen3:1.7b"


class TestSignalReport:
    def test_report_is_frozen(self) -> None:
        report = SignalReport(
            model=None,
            n_tasks=1,
            group_size=2,
            signal_rate=0.0,
            signal_rate_ci95=0.0,
            dead_too_easy=1,
            dead_too_hard=0,
            mean_spread=0.0,
        )
        with pytest.raises(Exception):
            report.n_tasks = 999  # type: ignore[misc]
