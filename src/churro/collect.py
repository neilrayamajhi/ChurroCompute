from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from churro.schema import (
    CHURRO_VERSION,
    EnvId,
    GroupId,
    ModelId,
    Rollout,
    TaskId,
    make_group_id,
)


class CollectError(Exception):
    pass


def collect_run(run_dir: Path) -> list[Rollout]:
    metadata_path = run_dir / "metadata.json"
    results_path = run_dir / "results.jsonl"
    if not metadata_path.is_file():
        raise CollectError(f"missing metadata.json in {run_dir}")
    if not results_path.is_file():
        raise CollectError(f"missing results.jsonl in {run_dir}")

    metadata = json.loads(metadata_path.read_text())
    env_id = EnvId(metadata["env_id"])
    model = ModelId(metadata["model"])
    collected_at = datetime.now(tz=UTC).isoformat()

    return list(_iter_rollouts(results_path, env_id, model, collected_at))


def write_rollouts(rollouts: Iterable[Rollout], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r.to_json_dict()) + "\n")
            n += 1
    return n


def count_timed_out(run_dir: Path) -> int:
    return sum(1 for row in _read_rows(run_dir / "results.jsonl") if _timed_out(row))


def merge_rollouts(
    existing: Iterable[Rollout], new: Iterable[Rollout]
) -> list[Rollout]:
    seen: set[tuple[str, str, int]] = set()
    merged: list[Rollout] = []
    for r in [*existing, *new]:
        key = (r.group_id, r.model, r.rollout_idx)
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)
    return merged


def load_rollouts(path: Path) -> list[Rollout]:
    if not path.is_file():
        raise CollectError(f"missing normalized file: {path}")
    rollouts: list[Rollout] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rollouts.append(
                Rollout(
                    env_id=EnvId(row["env_id"]),
                    task_id=TaskId(row["task_id"]),
                    group_id=GroupId(row["group_id"]),
                    rollout_idx=int(row["rollout_idx"]),
                    model=ModelId(row["model"]),
                    reward=float(row["reward"]),
                    num_turns=int(row["num_turns"]),
                    completion_tokens=int(row["completion_tokens"]),
                    collected_at=row["collected_at"],
                    churro_version=row["churro_version"],
                )
            )
    return rollouts


def _iter_rollouts(
    results_path: Path,
    env_id: EnvId,
    model: ModelId,
    collected_at: str,
) -> Iterable[Rollout]:
    seen: Counter[TaskId] = Counter()
    for row in _read_rows(results_path):
        # A rollout cut off by vf-eval's timeout never answered, so it is not
        # evidence the model failed; scoring it 0 would inflate dead_too_hard.
        if _timed_out(row):
            continue
        task_id = TaskId(str(row["example_id"]))
        rollout_idx = seen[task_id]
        seen[task_id] += 1
        yield Rollout(
            env_id=env_id,
            task_id=task_id,
            group_id=make_group_id(env_id, task_id, model),
            rollout_idx=rollout_idx,
            model=model,
            reward=float(row["reward"]),
            num_turns=int(row.get("num_turns", row["metrics"]["num_turns"])),
            completion_tokens=int(row["token_usage"]["output_tokens"]),
            collected_at=collected_at,
            churro_version=CHURRO_VERSION,
        )


def _read_rows(results_path: Path) -> Iterable[dict[str, object]]:
    with results_path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _timed_out(row: dict[str, object]) -> bool:
    return row.get("stop_condition") == "timeout_reached"
