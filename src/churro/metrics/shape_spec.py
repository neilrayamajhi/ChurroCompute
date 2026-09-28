from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from churro.metrics.shape import ShapeReport, reward_shape
from churro.schema import (
    CHURRO_VERSION,
    EnvId,
    ModelId,
    Rollout,
    TaskId,
    make_group_id,
)


def _rollout(reward: float, *, model: str = "qwen3:1.7b", task: str = "t") -> Rollout:
    env, task_id, model_id = EnvId("env"), TaskId(task), ModelId(model)
    return Rollout(
        env_id=env,
        task_id=task_id,
        group_id=make_group_id(env, task_id, model_id),
        rollout_idx=0,
        model=model_id,
        reward=reward,
        num_turns=1,
        completion_tokens=10,
        collected_at="2026-09-28T00:00:00+00:00",
        churro_version=CHURRO_VERSION,
    )


def _rollouts(rewards: list[float], model: str = "qwen3:1.7b") -> list[Rollout]:
    return [_rollout(r, model=model) for r in rewards]


class TestRewardShape:
    def test_returns_none_when_no_rollouts(self) -> None:
        assert reward_shape([]) is None

    def test_even_split_of_zero_and_one_is_two_effective_bins(self) -> None:
        report = reward_shape(_rollouts([0.0, 1.0, 0.0, 1.0]))

        assert report == ShapeReport(
            model=None,
            n_rollouts=4,
            distinct_values=2,
            effective_bins=2.0,
            effectively_binary=True,
            frac_at_extremes=1.0,
            histogram={0.0: 2, 1.0: 2},
        )

    def test_uniform_over_four_values_is_four_bins_and_not_binary(self) -> None:
        report = reward_shape(_rollouts([0.0, 0.25, 0.5, 0.75] * 3))

        assert report is not None
        assert (report.effective_bins, report.effectively_binary) == (4.0, False)

    def test_many_rare_values_around_one_dominant_value_is_effectively_binary(
        self,
    ) -> None:
        rare = [round(0.05 * i, 2) for i in range(1, 12)]
        report = reward_shape(_rollouts([1.0] * 500 + rare))

        assert report is not None
        assert (report.distinct_values, report.effectively_binary) == (12, True)

    def test_frac_at_extremes_counts_only_exact_zero_and_one(self) -> None:
        report = reward_shape(_rollouts([0.0, 0.5, 0.5, 1.0]))

        assert report is not None
        assert report.frac_at_extremes == 0.5

    def test_filters_to_one_model_when_given(self) -> None:
        mixed = _rollouts([0.0, 1.0], model="qwen3:0.6b") + _rollouts(
            [0.5, 0.5], model="qwen3:14b"
        )

        report = reward_shape(mixed, model=ModelId("qwen3:14b"))

        assert report is not None
        assert (report.model, report.n_rollouts, report.histogram) == (
            "qwen3:14b",
            2,
            {0.5: 2},
        )

    @given(st.lists(st.sampled_from([0.0, 0.1, 0.5, 0.9, 1.0]), min_size=1))
    def test_effective_bins_is_between_one_and_distinct_values(
        self, rewards: list[float]
    ) -> None:
        report = reward_shape(_rollouts(rewards))

        assert report is not None
        assert 1.0 <= report.effective_bins <= report.distinct_values + 1e-9

    @given(st.lists(st.floats(0, 1, allow_nan=False), min_size=1))
    def test_histogram_counts_add_up_to_rollouts(self, rewards: list[float]) -> None:
        report = reward_shape(_rollouts(rewards))

        assert report is not None
        assert sum(report.histogram.values()) == report.n_rollouts == len(rewards)
