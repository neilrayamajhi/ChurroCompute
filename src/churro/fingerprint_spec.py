from __future__ import annotations

import json

import pytest

from churro.fingerprint import FingerprintError, build_fingerprint
from churro.schema import (
    CHURRO_VERSION,
    EnvId,
    ModelId,
    Rollout,
    TaskId,
    make_group_id,
)

LADDER = (ModelId("qwen3:0.6b"), ModelId("qwen3:1.7b"))
LADDER_VERSION = "test-ladder"
GENERATED_AT = "2026-09-28T00:00:00+00:00"
ENV_VERSION = "0.1.3"


def _rollout(model: str, task: str, idx: int, reward: float) -> Rollout:
    env, task_id, model_id = EnvId("toy-env"), TaskId(task), ModelId(model)
    return Rollout(
        env_id=env,
        task_id=task_id,
        group_id=make_group_id(env, task_id, model_id),
        rollout_idx=idx,
        model=model_id,
        reward=reward,
        num_turns=1,
        completion_tokens=10,
        collected_at=GENERATED_AT,
        churro_version=CHURRO_VERSION,
    )


def _ladder_run() -> list[Rollout]:
    small = [_rollout("qwen3:0.6b", t, i, r) for t, rs in {"a": [0, 1], "b": [0, 0]}.items() for i, r in enumerate(rs)]
    large = [_rollout("qwen3:1.7b", t, i, r) for t, rs in {"a": [1, 1], "b": [1, 0]}.items() for i, r in enumerate(rs)]
    return small + large


def _fingerprint(rollouts: list[Rollout], grader_check: str | None = "ok") -> dict[str, object]:
    return build_fingerprint(
        rollouts,
        ladder=LADDER,
        ladder_version=LADDER_VERSION,
        env_version=ENV_VERSION,
        grader_check=grader_check,
        generated_at=GENERATED_AT,
    )


class TestBuildFingerprint:
    def test_refuses_to_fingerprint_no_rollouts(self) -> None:
        with pytest.raises(FingerprintError):
            _fingerprint([])

    def test_refuses_to_publish_a_fingerprint_of_zeros(self) -> None:
        zeros = [_rollout("qwen3:0.6b", "a", i, 0.0) for i in range(4)]

        with pytest.raises(FingerprintError):
            _fingerprint(zeros)

    def test_records_every_version_needed_to_compare_fingerprints(self) -> None:
        fp = _fingerprint(_ladder_run())

        assert {k: fp[k] for k in (
            "env_id", "churro_version", "ladder_version", "env_version",
            "generated_at", "group_size", "n_rollouts", "grader_check",
        )} == {
            "env_id": "toy-env",
            "churro_version": CHURRO_VERSION,
            "ladder_version": LADDER_VERSION,
            "env_version": ENV_VERSION,
            "generated_at": GENERATED_AT,
            "group_size": 2,
            "n_rollouts": 8,
            "grader_check": "ok",
        }

    def test_includes_difficulty_across_the_ladder(self) -> None:
        fp = _fingerprint(_ladder_run())
        difficulty = fp["difficulty"]

        assert isinstance(difficulty, dict)
        assert [r["model"] for r in difficulty["rungs"]] == list(LADDER)

    def test_includes_reward_shape_overall_and_per_model(self) -> None:
        fp = _fingerprint(_ladder_run())
        shape = fp["shape"]

        assert isinstance(shape, dict)
        assert sorted(shape) == ["overall", "qwen3:0.6b", "qwen3:1.7b"]

    def test_is_json_serialisable(self) -> None:
        fp = _fingerprint(_ladder_run())

        assert json.loads(json.dumps(fp)) == fp
