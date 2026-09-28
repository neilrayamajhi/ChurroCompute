from __future__ import annotations

import asyncio

import verifiers as vf
from verifiers.types import AssistantMessage, State


def score_text(
    env: vf.Environment, row: dict[str, object], text: str, as_dict: bool = False
) -> float:
    # Build the state the way a real rollout does: the completion is a list of
    # AssistantMessage objects, not dicts, and row columns arrive as-is.
    state = State(
        input=dict(row),
        prompt=row["prompt"],
        completion=[
            {"role": "assistant", "content": text}
            if as_dict
            else AssistantMessage(content=text)
        ],
        answer=row.get("answer", ""),
        info=row.get("info") or {},
        trajectory=[],
    )
    asyncio.run(env.rubric.score_group([state]))
    return float(state["reward"])
