from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NewType

EnvId = NewType("EnvId", str)
TaskId = NewType("TaskId", str)
ModelId = NewType("ModelId", str)
GroupId = NewType("GroupId", str)

CHURRO_VERSION = "0.1.0"


def make_group_id(env_id: EnvId, task_id: TaskId, model: ModelId) -> GroupId:
    return GroupId(f"{env_id}::{task_id}::{model}")


@dataclass(frozen=True, slots=True)
class Rollout:
    env_id: EnvId
    task_id: TaskId
    group_id: GroupId
    rollout_idx: int
    model: ModelId
    reward: float
    num_turns: int
    completion_tokens: int
    collected_at: str
    churro_version: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)
