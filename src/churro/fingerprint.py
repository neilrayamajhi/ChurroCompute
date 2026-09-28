from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict

from churro.metrics.difficulty import difficulty_curve
from churro.metrics.shape import ShapeReport, reward_shape
from churro.schema import CHURRO_VERSION, ModelId, Rollout


class FingerprintError(Exception):
    pass


def build_fingerprint(
    rollouts: Sequence[Rollout],
    *,
    ladder: Sequence[ModelId],
    ladder_version: str,
    env_version: str | None,
    grader_check: str | None,
    generated_at: str,
) -> dict[str, object]:
    if not rollouts:
        raise FingerprintError("no rollouts to fingerprint")
    if all(r.reward == 0.0 for r in rollouts):
        # Non-negotiable: never publish a fingerprint of zeros. Every reward
        # being 0 far more often means a broken grader than a hard env.
        raise FingerprintError(
            f"every one of {len(rollouts)} rewards is 0; check the grader before publishing"
        )

    difficulty = difficulty_curve(rollouts, ladder, ladder_version)
    assert difficulty is not None
    group_sizes = Counter(r.group_id for r in rollouts).values()
    models_present = [m for m in ladder if any(r.model == m for r in rollouts)]
    shape = {"overall": _shape_json(reward_shape(rollouts))}
    shape.update({m: _shape_json(reward_shape(rollouts, model=m)) for m in models_present})

    return {
        "env_id": difficulty.env_id,
        "churro_version": CHURRO_VERSION,
        "ladder_version": ladder_version,
        "env_version": env_version,
        "generated_at": generated_at,
        "group_size": statistics.mode(group_sizes),
        "n_rollouts": len(rollouts),
        "grader_check": grader_check,
        "difficulty": difficulty.to_json_dict(),
        "shape": shape,
    }


def _shape_json(report: ShapeReport | None) -> dict[str, object] | None:
    if report is None:
        return None
    data = asdict(report)
    data["histogram"] = {f"{k:g}": v for k, v in report.histogram.items()}
    return data
