from __future__ import annotations

import json

import pytest

from churro.metrics.difficulty import (
    DifficultyReport,
    RungReport,
    difficulty_curve,
)
from churro.schema import (
    CHURRO_VERSION,
    EnvId,
    ModelId,
    Rollout,
    TaskId,
    make_group_id,
)


def _rollout(*, task: str, reward: float, model: str, env: str = "gsm8k") -> Rollout:
    env_id = EnvId(env)
    task_id = TaskId(task)
    model_id = ModelId(model)
    return Rollout(
        env_id=env_id,
        task_id=task_id,
        group_id=make_group_id(env_id, task_id, model_id),
        rollout_idx=0,
        model=model_id,
        reward=reward,
        num_turns=1,
        completion_tokens=100,
        collected_at="2026-09-09T00:00:00+00:00",
        churro_version=CHURRO_VERSION,
    )


def _group(task: str, scores: list[float], model: str, env: str = "gsm8k") -> list[Rollout]:
    return [_rollout(task=task, reward=s, model=model, env=env) for s in scores]


_LADDER = (
    ModelId("qwen3:0.6b"),
    ModelId("qwen3:1.7b"),
    ModelId("qwen3:4b"),
)


class TestDifficultyCurve:
    def test_returns_none_when_no_rollouts(self) -> None:
        assert difficulty_curve([], _LADDER, "qwen3-v1") is None

    def test_missing_rung_has_none_signal_but_is_kept(self) -> None:
        # only middle rung has data; outer rungs must still appear in the report
        rollouts = _group("t1", [1.0, 0.0], model="qwen3:1.7b")
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert tuple(r.model for r in report.rungs) == _LADDER
        assert report.rungs[0].signal is None
        assert report.rungs[1].signal is not None
        assert report.rungs[2].signal is None

    def test_pass_rate_and_mean_reward_are_per_rung(self) -> None:
        # 3 of 4 pass at 1.7b, 0 of 4 pass at 0.6b
        rollouts = [
            *_group("t1", [1.0, 1.0, 1.0, 0.0], model="qwen3:1.7b"),
            *_group("t1", [0.0, 0.0, 0.0, 0.0], model="qwen3:0.6b"),
        ]
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        by_model = {r.model: r for r in report.rungs}
        assert by_model[ModelId("qwen3:1.7b")].pass_rate == 0.75
        assert by_model[ModelId("qwen3:1.7b")].mean_reward == 0.75
        assert by_model[ModelId("qwen3:0.6b")].pass_rate == 0.0
        assert by_model[ModelId("qwen3:0.6b")].mean_reward == 0.0

    def test_best_signal_model_is_max_signal_rate_rung(self) -> None:
        # 1.7b has live tasks (signal>0); 0.6b is all-fail (signal=0); 4b is all-pass (signal=0)
        rollouts = [
            *_group("t1", [0.0, 0.0], model="qwen3:0.6b"),
            *_group("t2", [0.0, 0.0], model="qwen3:0.6b"),
            *_group("t1", [1.0, 0.0], model="qwen3:1.7b"),
            *_group("t2", [1.0, 0.0], model="qwen3:1.7b"),
            *_group("t1", [1.0, 1.0], model="qwen3:4b"),
            *_group("t2", [1.0, 1.0], model="qwen3:4b"),
        ]
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert report.best_signal_model == ModelId("qwen3:1.7b")

    def test_floor_is_weakest_rung_with_any_pass(self) -> None:
        rollouts = [
            *_group("t1", [0.0, 0.0], model="qwen3:0.6b"),  # 0% pass
            *_group("t1", [1.0, 0.0], model="qwen3:1.7b"),  # 50% pass — this is the floor
            *_group("t1", [1.0, 1.0], model="qwen3:4b"),  # 100% pass
        ]
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert report.floor == ModelId("qwen3:1.7b")

    def test_ceiling_is_strongest_rung_not_fully_saturated(self) -> None:
        rollouts = [
            *_group("t1", [0.0, 0.0], model="qwen3:0.6b"),
            *_group("t1", [1.0, 0.0], model="qwen3:1.7b"),  # 50% — ceiling (last not-full)
            *_group("t1", [1.0, 1.0], model="qwen3:4b"),  # 100% — saturated
        ]
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert report.ceiling == ModelId("qwen3:1.7b")

    def test_slope_is_pass_rate_delta_over_rung_distance(self) -> None:
        # floor=1.7b (0.5 pass), ceiling=4b (0.75 pass), distance=1 rung → slope=0.25
        rollouts = [
            *_group("t1", [1.0, 0.0], model="qwen3:1.7b"),  # 0.5
            *_group("t1", [1.0, 1.0, 1.0, 0.0], model="qwen3:4b"),  # 0.75
        ]
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert report.slope == 0.25

    def test_saturated_flag_true_when_ceiling_dominates_dead_too_easy(self) -> None:
        # 4b: 5 tasks, all-pass groups → dead_too_easy = 5/5 = 100%
        rollouts = []
        for i in range(5):
            rollouts.extend(_group(f"t{i}", [1.0, 1.0], model="qwen3:4b"))
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert report.saturated is True

    def test_possibly_broken_when_every_rung_zero_signal_and_zero_pass(self) -> None:
        rollouts = [
            *_group("t1", [0.0, 0.0], model="qwen3:0.6b"),
            *_group("t1", [0.0, 0.0], model="qwen3:1.7b"),
            *_group("t1", [0.0, 0.0], model="qwen3:4b"),
        ]
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert report.possibly_broken is True
        assert report.best_signal_model is None

    def test_mixed_env_id_raises(self) -> None:
        rollouts = [
            *_group("t1", [1.0, 0.0], model="qwen3:1.7b", env="gsm8k"),
            *_group("t1", [1.0, 0.0], model="qwen3:1.7b", env="regex-craft"),
        ]
        with pytest.raises(ValueError):
            difficulty_curve(rollouts, _LADDER, "qwen3-v1")

    def test_env_id_and_ladder_version_stamped_on_report(self) -> None:
        rollouts = _group("t1", [1.0, 0.0], model="qwen3:1.7b", env="gsm8k")
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        assert report.env_id == EnvId("gsm8k")
        assert report.ladder_version == "qwen3-v1"

    def test_to_json_dict_round_trips(self) -> None:
        rollouts = _group("t1", [1.0, 0.0], model="qwen3:1.7b")
        report = difficulty_curve(rollouts, _LADDER, "qwen3-v1")
        assert report is not None
        as_json = json.dumps(report.to_json_dict())
        assert "qwen3-v1" in as_json
        assert "gsm8k" in as_json


class TestReportsFrozen:
    def test_rung_report_is_frozen(self) -> None:
        rung = RungReport(
            model=ModelId("qwen3:1.7b"),
            signal=None,
            pass_rate=0.0,
            mean_reward=0.0,
        )
        with pytest.raises(Exception):
            rung.pass_rate = 1.0  # type: ignore[misc]

    def test_difficulty_report_is_frozen(self) -> None:
        report = DifficultyReport(
            env_id=EnvId("gsm8k"),
            ladder_version="qwen3-v1",
            rungs=(),
            floor=None,
            ceiling=None,
            best_signal_model=None,
            slope=0.0,
            saturated=False,
            possibly_broken=False,
        )
        with pytest.raises(Exception):
            report.slope = 1.0  # type: ignore[misc]
