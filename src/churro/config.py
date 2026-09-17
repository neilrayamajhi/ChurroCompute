from __future__ import annotations

from churro.schema import ModelId

LADDER: tuple[ModelId, ...] = tuple(
    ModelId(m)
    for m in (
        "qwen3:0.6b",
        "qwen3:1.7b",
        "qwen3:4b",
        "qwen3:8b",
        "qwen3:14b",
    )
)
LADDER_VERSION = "qwen3-v1"
