from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from churro.hub_scan import classify_check, is_scan_candidate
from churro.pod_ladder import PRIME_HUB_INDEX

PRIME_CLI = ["uvx", "--from", "prime==0.6.28", "prime"]
PAGE_SIZE = 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight the graders of the most-starred single-turn Prime Hub envs"
    )
    parser.add_argument("--limit", type=int, default=40, help="candidates to check this run")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per env")
    parser.add_argument("--out", type=Path, default=Path("outputs/hub_scan.jsonl"))
    args = parser.parse_args()

    done = _already_scanned(args.out)
    todo = [e for e in _candidates() if e["environment"] not in done][: args.limit]
    print(f"{len(done)} already scanned; checking {len(todo)} more")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i, env in enumerate(todo, 1):
        outcome = _check(env["environment"], args.timeout)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**env, "outcome": outcome}) + "\n")
        print(f"[{i}/{len(todo)}] {env['environment']:48} {outcome}", flush=True)

    totals = Counter(json.loads(line)["outcome"] for line in args.out.open(encoding="utf-8"))
    print("totals:", dict(totals))


def _candidates() -> list[dict[str, object]]:
    envs: list[dict[str, object]] = []
    page = 1
    while True:
        listing = json.loads(
            subprocess.run(
                [*PRIME_CLI, "env", "list", "--visibility", "PUBLIC", "--sort", "stars",
                 "--order", "desc", "-n", str(PAGE_SIZE), "-p", str(page), "--output", "json"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                check=True, env=_env(),
            ).stdout
        )
        envs += listing["environments"]
        if page * PAGE_SIZE >= listing["total"]:
            break
        page += 1
    # The listing can repeat an env across pages while stars shift under it.
    seen: set[object] = set()
    unique = []
    for e in envs:
        if e["environment"] not in seen:
            seen.add(e["environment"])
            unique.append(e)
    return [e for e in unique if is_scan_candidate(list(e.get("tags") or []))]


def _check(env_slug: str, timeout: int) -> str:
    env_name = env_slug.rsplit("/", 1)[-1]
    try:
        done = subprocess.run(
            ["uv", "run", "--with", env_name,
             "--extra-index-url", PRIME_HUB_INDEX.format(slug=env_slug),
             "python", "tools/check_grader.py", env_name, "--rows", "3"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=_env(),
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    return classify_check(done.returncode, done.stdout, done.stderr)


def _already_scanned(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(line)["environment"] for line in path.open(encoding="utf-8") if line.strip()}


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONUTF8": "1", "PRIME_DISABLE_VERSION_CHECK": "1"}


if __name__ == "__main__":
    sys.exit(main())
