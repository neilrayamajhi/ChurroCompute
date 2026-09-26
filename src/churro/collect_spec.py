from __future__ import annotations

import json
from pathlib import Path

import pytest

from churro.collect import (
    CollectError,
    collect_run,
    merge_rollouts,
)
from churro.schema import (
    CHURRO_VERSION,
    EnvId,
    ModelId,
    Rollout,
    TaskId,
    make_group_id,
)


def _write_run(
    tmp_path: Path,
    *,
    env_id: str = "gsm8k",
    model: str = "qwen3:1.7b",
    rollouts: list[dict[str, object]] | None = None,
) -> Path:
    run_dir = tmp_path / f"{env_id}--{model.replace(':', '_')}" / "abc12345"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": env_id,
                "model": model,
                "num_examples": 3,
                "rollouts_per_example": 2,
                "version_info": {"vf_version": "0.3.0", "env_version": "0.1.3"},
            }
        )
    )
    if rollouts is None:
        rollouts = [
            _fake_rollout(example_id=1, reward=1.0),
            _fake_rollout(example_id=1, reward=0.0),
            _fake_rollout(example_id=2, reward=1.0),
            _fake_rollout(example_id=2, reward=1.0),
            _fake_rollout(example_id=0, reward=0.0),
            _fake_rollout(example_id=0, reward=1.0),
        ]
    with (run_dir / "results.jsonl").open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")
    return run_dir


def _fake_rollout(*, example_id: int, reward: float) -> dict[str, object]:
    return {
        "example_id": example_id,
        "reward": reward,
        "metrics": {"correct_answer": reward, "num_turns": 1.0},
        "token_usage": {"input_tokens": 56, "output_tokens": 653},
        "num_turns": 1,
    }


describe = "collect_run"


class TestCollectRun:
    def test_returns_one_rollout_per_results_line(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        result = collect_run(run_dir)
        assert len(result) == 6

    def test_maps_metadata_and_reward_onto_every_rollout(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        rollouts = collect_run(run_dir)
        first = rollouts[0]
        assert first.env_id == "gsm8k"
        assert first.model == "qwen3:1.7b"
        assert first.task_id == "1"
        assert first.reward == 1.0
        assert first.num_turns == 1
        assert first.completion_tokens == 653
        assert first.churro_version == CHURRO_VERSION

    def test_group_id_is_env_task_model_joined_by_double_colon(
        self, tmp_path: Path
    ) -> None:
        run_dir = _write_run(tmp_path)
        rollouts = collect_run(run_dir)
        assert rollouts[0].group_id == "gsm8k::1::qwen3:1.7b"

    def test_rollout_idx_counts_up_per_task_in_file_order(
        self, tmp_path: Path
    ) -> None:
        run_dir = _write_run(tmp_path)
        rollouts = collect_run(run_dir)
        by_task = [(r.task_id, r.rollout_idx) for r in rollouts]
        assert by_task == [
            ("1", 0),
            ("1", 1),
            ("2", 0),
            ("2", 1),
            ("0", 0),
            ("0", 1),
        ]

    def test_missing_metadata_raises_collect_error(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        (run_dir / "metadata.json").unlink()
        with pytest.raises(CollectError):
            collect_run(run_dir)

    def test_missing_results_raises_collect_error(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        (run_dir / "results.jsonl").unlink()
        with pytest.raises(CollectError):
            collect_run(run_dir)

    def test_collected_at_is_iso_utc_timestamp(self, tmp_path: Path) -> None:
        from datetime import datetime

        run_dir = _write_run(tmp_path)
        rollouts = collect_run(run_dir)
        parsed = datetime.fromisoformat(rollouts[0].collected_at)
        assert parsed.tzinfo is not None


def _stored(*, task: str, idx: int, reward: float, model: str = "qwen3:1.7b") -> Rollout:
    env = EnvId("regex-craft")
    task_id = TaskId(task)
    model_id = ModelId(model)
    return Rollout(
        env_id=env,
        task_id=task_id,
        group_id=make_group_id(env, task_id, model_id),
        rollout_idx=idx,
        model=model_id,
        reward=reward,
        num_turns=1,
        completion_tokens=100,
        collected_at="2026-09-25T00:00:00+00:00",
        churro_version=CHURRO_VERSION,
    )


class TestMergeRollouts:
    def test_appends_new_rollouts_after_existing_ones(self) -> None:
        existing = [_stored(task="0", idx=0, reward=1.0)]
        new = [_stored(task="1", idx=0, reward=0.0)]

        assert merge_rollouts(existing, new) == existing + new

    def test_keeps_existing_copy_when_new_one_has_same_identity(self) -> None:
        existing = [_stored(task="0", idx=0, reward=1.0)]
        rerun = [_stored(task="0", idx=0, reward=0.0)]

        assert merge_rollouts(existing, rerun) == existing

    def test_same_task_and_index_on_different_models_are_distinct(self) -> None:
        small = _stored(task="0", idx=0, reward=1.0, model="qwen3:0.6b")
        large = _stored(task="0", idx=0, reward=1.0, model="qwen3:14b")

        assert merge_rollouts([small], [large]) == [small, large]

    def test_merging_twice_is_the_same_as_merging_once(self) -> None:
        existing = [_stored(task="0", idx=0, reward=1.0)]
        new = [_stored(task="0", idx=1, reward=0.0)]
        once = merge_rollouts(existing, new)

        assert merge_rollouts(once, new) == once

