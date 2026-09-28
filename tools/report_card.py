from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from churro.collect import load_rollouts
from churro.config import LADDER, LADDER_VERSION
from churro.fingerprint import FingerprintError, build_fingerprint

FINGERPRINTS = Path("data/fingerprints")
HUB_SCAN = Path("outputs/hub_scan.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an env's fingerprint (report card) from its normalized rollouts"
    )
    parser.add_argument("data_path", type=Path, help="normalized data/raw/<env>.jsonl")
    parser.add_argument("--env-version", default=None, help="env package version, if known")
    args = parser.parse_args()

    rollouts = load_rollouts(args.data_path)
    env_name = args.data_path.stem
    try:
        fp = build_fingerprint(
            rollouts,
            ladder=LADDER,
            ladder_version=LADDER_VERSION,
            env_version=args.env_version or _env_version(env_name),
            grader_check=_grader_check(env_name),
            generated_at=datetime.now(tz=UTC).isoformat(),
        )
    except FingerprintError as e:
        sys.exit(f"Not writing a report card for {env_name}: {e}")

    FINGERPRINTS.mkdir(parents=True, exist_ok=True)
    out = FINGERPRINTS / f"{env_name}.json"
    out.write_text(json.dumps(fp, indent=2), encoding="utf-8")
    print(_card(fp))
    print(f"\nsaved {out}")


def _card(fp: dict) -> str:
    d = fp["difficulty"]
    lines = [
        f"REPORT CARD: {fp['env_id']}  (env {fp['env_version'] or '?'}, ladder {fp['ladder_version']}, "
        f"churro {fp['churro_version']}, group size {fp['group_size']})",
        f"answer-key check: {fp['grader_check'] or 'not run'}",
        "",
        f"{'model':<13}{'tasks':>6}{'pass':>7}{'signal':>14}{'easy':>6}{'hard':>6}{'eff_bins':>10}",
    ]
    for rung in d["rungs"]:
        s, shape = rung["signal"], fp["shape"].get(rung["model"])
        if s is None:
            lines.append(f"{rung['model']:<13}{'(no data)':>6}")
            continue
        signal = f"{s['signal_rate']:.0%} ±{s['signal_rate_ci95']:.0%}"
        bins = f"{shape['effective_bins']:.2f}" if shape else "-"
        lines.append(
            f"{rung['model']:<13}{s['n_tasks']:>6}{rung['pass_rate']:>7.0%}{signal:>14}"
            f"{s['dead_too_easy']:>6}{s['dead_too_hard']:>6}{bins:>10}"
        )
    overall = fp["shape"]["overall"]
    lines += [
        "",
        f"best model to train: {d['best_signal_model'] or '-'}   floor: {d['floor'] or '-'}   "
        f"ceiling: {d['ceiling'] or '-'}",
        f"saturated: {d['saturated']}   possibly broken: {d['possibly_broken']}   "
        f"effectively binary reward: {overall['effectively_binary']} ({overall['effective_bins']:.2f} bins)",
    ]
    return "\n".join(lines)


def _grader_check(env_name: str) -> str | None:
    if not HUB_SCAN.is_file():
        return None
    for line in HUB_SCAN.open(encoding="utf-8"):
        row = json.loads(line)
        if row["environment"].rsplit("/", 1)[-1] == env_name:
            return row["outcome"]
    return None


def _env_version(env_name: str) -> str | None:
    for metadata in sorted(Path("outputs/runpod").glob(f"*/outputs/evals/{env_name}--*/*/metadata.json")):
        version = json.loads(metadata.read_text()).get("version_info", {}).get("env_version")
        if version:
            return str(version)
    return None


if __name__ == "__main__":
    main()
