from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from functools import cache
from pathlib import Path

from churro.config import LADDER
from churro.pod_ladder import (
    POD_WORKDIR,
    PRIME_HUB_INDEX,
    Cloud,
    GpuOffer,
    LadderPlan,
    parallel_requests_for,
    parse_status,
    rank_gpu_offers,
    render_pod_script,
)

# 48GB cards first: they serve 8 requests at once (parallel_requests_for);
# 24GB cards fall back to 4 so qwen3:14b's KV cache still fits.
GPU_PREFERENCES: list[tuple[str, Cloud]] = [
    ("NVIDIA RTX A6000", "COMMUNITY"),
    ("NVIDIA A40", "SECURE"),
    ("NVIDIA RTX A6000", "SECURE"),
    ("NVIDIA L40", "COMMUNITY"),
    ("NVIDIA RTX 6000 Ada Generation", "COMMUNITY"),
    ("NVIDIA GeForce RTX 4090", "COMMUNITY"),
    ("NVIDIA GeForce RTX 3090", "SECURE"),
    ("NVIDIA GeForce RTX 4090", "SECURE"),
]
POD_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
BUDGET_LIMIT_USD = 3.0
POLL_SECONDS = 60
# The pod outlives this machine's deadline so the final snapshot can still be
# downloaded; with equal deadlines the pod deleted itself first and took the
# last model's results with it.
SELF_DESTRUCT_GRACE_S = 15 * 60
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
RESULTS_ROOT = Path("outputs/runpod")
STATE_FILE = RESULTS_ROOT / "active.json"
FINISHED_STATES = {"done", "timeout", "failed"}
UPLOAD_PATHS = ["pyproject.toml", "uv.lock", "src", "tools"]


class LadderRunError(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rent a RunPod GPU, run the qwen3 ladder on one environment, bring results home, delete the pod."
    )
    parser.add_argument("env_slug", help="Prime Hub environment, e.g. liuliu/regex-craft")
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--rollouts-per-example", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=2.5)
    parser.add_argument("--max-price-per-hr", type=float, default=0.80)
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument(
        "--skip-grader-check",
        action="store_true",
        help="rent a GPU even if the env's grader fails the pre-flight check",
    )
    args = parser.parse_args()

    if not os.environ.get("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY is not set. Save it with setx and reopen your terminal.")
    for tool in ("runpodctl", "ssh", "scp", "tar", "ssh-keygen"):
        if shutil.which(tool) is None:
            sys.exit(f"missing required tool on PATH: {tool}")

    plan = LadderPlan(
        env_slug=args.env_slug,
        ladder=LADDER,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        per_model_timeout_s=45 * 60,
        per_rollout_timeout_s=10 * 60,
        self_destruct_s=int(args.max_hours * 3600) + SELF_DESTRUCT_GRACE_S,
        parallel=parallel_requests_for(24),
    )

    pod_id = _resume_pod_id()
    _keep_pc_awake(True)
    try:
        if pod_id:
            print(f"Reconnecting to pod {pod_id} from a previous run.")
        else:
            if not args.skip_grader_check:
                _require_working_grader(plan.env_slug)
            _ensure_ssh_key()
            offers = rank_gpu_offers(_runpodctl("gpu", "list"), GPU_PREFERENCES, args.max_price_per_hr)
            if not offers:
                sys.exit("No suitable GPU is available right now. Nothing was rented. Try again later.")
            _confirm_cost(offers, args.max_hours, args.yes)
            pod_id, offer = _create_pod(offers, plan.env_slug)
            _save_state(pod_id)
            plan = replace(plan, parallel=parallel_requests_for(offer.memory_gb))
            print(f"Running {plan.parallel} requests at once on this {offer.memory_gb}GB card.")
            _start_ladder(pod_id, plan)
        if not _wait_for_ladder(pod_id, args.max_hours):
            raise LadderRunError(f"the run did not finish within {args.max_hours:g}h")
    except LadderRunError as e:
        print(f"\nRun stopped: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopped by you.", file=sys.stderr)
    finally:
        if pod_id:
            _download_snapshot(pod_id)
            _delete_pod(pod_id)
            _collect_and_report(RESULTS_ROOT / pod_id)
        _warn_about_running_pods()
        _keep_pc_awake(False)


def _keep_pc_awake(on: bool) -> None:
    # Snapshots only download while this PC is awake; if it sleeps, the pod can
    # self-destruct with results that never came home. Closing a laptop lid
    # still forces sleep.
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    flags = es_continuous | (es_system_required if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    if on:
        print("Keeping this PC awake until the run ends (keep the laptop lid open).")


def _require_working_grader(env_slug: str) -> None:
    env_name = env_slug.rsplit("/", 1)[-1]
    print(f"Checking {env_name}'s grader before renting anything...")
    done = subprocess.run(
        [
            "uv", "run", "--with", env_name,
            "--extra-index-url", PRIME_HUB_INDEX.format(slug=env_slug),
            "python", "tools/check_grader.py", env_name,
        ],
        capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"},
    )
    verdict = [ln for ln in done.stdout.splitlines() if ln.startswith("grader check")]
    print(f"  {verdict[-1] if verdict else done.stderr.strip()[-300:]}")
    if done.returncode != 0:
        sys.exit(
            "Not renting a GPU: this env's grader would make every number meaningless.\n"
            "Use --skip-grader-check to run it anyway (e.g. to collect rollouts for re-scoring)."
        )


def _runpodctl(*args: str) -> object:
    done = subprocess.run(["runpodctl", *args, "-o", "json"], capture_output=True, text=True)
    if done.returncode != 0:
        raise LadderRunError(f"runpodctl {' '.join(args)} failed: {done.stderr.strip() or done.stdout.strip()}")
    return json.loads(done.stdout) if done.stdout.strip() else None


def _resume_pod_id() -> str | None:
    if not STATE_FILE.is_file():
        return None
    pod_id = json.loads(STATE_FILE.read_text())["pod_id"]
    alive = subprocess.run(["runpodctl", "pod", "get", pod_id], capture_output=True).returncode == 0
    if not alive:
        STATE_FILE.unlink()
        return None
    return pod_id


def _save_state(pod_id: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"pod_id": pod_id}))


def _ensure_ssh_key() -> None:
    public_key_path = SSH_KEY.with_suffix(".pub")
    if not SSH_KEY.is_file():
        print(f"Creating an SSH key at {SSH_KEY} (one time only).")
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(SSH_KEY)], check=True)
    public_key = public_key_path.read_text().split()[1]
    registered = json.dumps(_runpodctl("ssh", "list-keys"))
    if public_key not in registered:
        print("Registering your SSH key with RunPod (one time only).")
        _runpodctl("ssh", "add-key", "--key-file", str(public_key_path))


def _confirm_cost(offers: list[GpuOffer], max_hours: float, skip_prompt: bool) -> None:
    worst = max(o.price_per_hr for o in offers) * (max_hours + SELF_DESTRUCT_GRACE_S / 3600)
    first = offers[0]
    print(
        f"Will try {first.gpu_id} ({first.cloud.lower()}) at ${first.price_per_hr:.2f}/hr first, "
        f"then {len(offers) - 1} fallback(s).\n"
        f"The pod deletes itself after {max_hours + SELF_DESTRUCT_GRACE_S / 3600:g}h at the latest: worst case ≈ ${worst:.2f}."
    )
    if worst > BUDGET_LIMIT_USD:
        sys.exit(f"Worst case is over the ${BUDGET_LIMIT_USD:.2f} limit. Lower --max-hours or --max-price-per-hr.")
    if not skip_prompt and input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled. Nothing was rented.")


def _create_pod(offers: list[GpuOffer], env_slug: str) -> tuple[str, GpuOffer]:
    for offer in offers:
        print(f"Renting {offer.gpu_id} ({offer.cloud.lower()})...")
        cmd = [
            "runpodctl", "pod", "create",
            "--name", f"churro-{env_slug.rsplit('/', 1)[-1]}",
            "--image", POD_IMAGE,
            "--gpu-id", offer.gpu_id,
            "--cloud-type", offer.cloud,
            "--container-disk-in-gb", "60",
            "--ports", "22/tcp",
            "--wait", "--wait-timeout", "10m",
            "-o", "json",
        ]
        if offer.cloud == "COMMUNITY":
            cmd.append("--public-ip")
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0:
            pod_id = json.loads(done.stdout)["id"]
            print(f"Pod {pod_id} is up.")
            return pod_id, offer
        stranded = re.search(r'"id"\s*:\s*"([a-z0-9]+)"', done.stderr + done.stdout)
        if stranded:
            _delete_pod(stranded.group(1))
        print(f"  not available ({(done.stderr or done.stdout).strip()[:200]}), trying the next option.")
    raise LadderRunError("none of the GPU options could be rented right now")


@cache
def _ssh_target(pod_id: str) -> tuple[str, str]:
    info = _runpodctl("ssh", "info", pod_id)
    if not isinstance(info, dict) or not info.get("ip") or not info.get("port"):
        raise LadderRunError(f"pod {pod_id} has no public SSH address yet: {info}")
    return f"root@{info['ip']}", str(info["port"])


def _ssh_options(port_flag: str, port: str) -> list[str]:
    return ["-i", str(SSH_KEY), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", port_flag, port]


def _ssh(pod_id: str, command: str) -> subprocess.CompletedProcess[str]:
    host, port = _ssh_target(pod_id)
    return subprocess.run(["ssh", *_ssh_options("-p", port), host, command], capture_output=True, text=True)


def _scp(pod_id: str, source: str, dest: str) -> None:
    host, port = _ssh_target(pod_id)
    done = subprocess.run(
        ["scp", *_ssh_options("-P", port), source.replace("POD:", f"{host}:"), dest.replace("POD:", f"{host}:")],
        capture_output=True, text=True,
    )
    if done.returncode != 0:
        raise LadderRunError(f"file copy failed: {done.stderr.strip()}")


def _start_ladder(pod_id: str, plan: LadderPlan) -> None:
    print("Uploading the Churro code and the run script...")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "churro.tgz"
        subprocess.run(["tar", "czf", str(bundle), "--exclude=__pycache__", *UPLOAD_PATHS], check=True)
        script = Path(tmp) / "run.sh"
        script.write_bytes(render_pod_script(plan).encode())
        _scp(pod_id, str(bundle), "POD:/root/churro.tgz")
        _scp(pod_id, str(script), "POD:/root/run.sh")
    launch = (
        f"mkdir -p {POD_WORKDIR} && tar xzf /root/churro.tgz -C {POD_WORKDIR} && "
        # Braces keep `&` on run.sh alone; otherwise the whole chain backgrounds
        # while still holding the SSH channel open until the run ends.
        f"{{ nohup setsid bash /root/run.sh > /dev/null 2>&1 < /dev/null & }}"
    )
    launched = _ssh(pod_id, launch).returncode == 0
    time.sleep(5)
    # Trust the pod's own status file over the SSH exit code: a dropped
    # connection must never be mistaken for a run that didn't start.
    if not launched and _ssh(pod_id, f"test -f {POD_WORKDIR}/status.tsv").returncode != 0:
        raise LadderRunError("could not start the run script on the pod")
    print("Run started on the pod. Checking in every minute (Ctrl+C stops and deletes the pod).")


def _wait_for_ladder(pod_id: str, max_hours: float) -> bool:
    deadline = time.monotonic() + max_hours * 3600
    last_seen: dict[str, str] = {}
    while time.monotonic() < deadline:
        try:
            done = _ssh(pod_id, f"cat {POD_WORKDIR}/status.tsv 2>/dev/null")
        except LadderRunError:
            done = None
        if done is not None and done.returncode == 0:
            status = parse_status(done.stdout)
            newly_changed = {
                step: state for step, state in status.steps.items() if last_seen.get(step) != state
            }
            for step, state in newly_changed.items():
                print(f"  {time.strftime('%H:%M')}  {step:<14} {state}")
            if any(state in FINISHED_STATES for state in newly_changed.values()):
                _download_snapshot(pod_id)
            last_seen = status.steps
            if status.finished:
                return True
        else:
            print(f"  {time.strftime('%H:%M')}  (couldn't reach the pod, will retry)")
        time.sleep(POLL_SECONDS)
    return False


def _download_snapshot(pod_id: str) -> None:
    dest = RESULTS_ROOT / pod_id
    dest.mkdir(parents=True, exist_ok=True)
    try:
        # Windows can't create paths containing ':', so rename inside the archive.
        _ssh(pod_id, f"cd {POD_WORKDIR} && tar czf results.tgz --transform 's/:/_/g' --ignore-failed-read outputs logs status.tsv")
        _scp(pod_id, f"POD:{POD_WORKDIR}/results.tgz", str(dest / "results.tgz"))
    except LadderRunError as e:
        print(f"  (could not download a snapshot: {e})")
        return
    subprocess.run(["tar", "xzf", str(dest / "results.tgz"), "-C", str(dest)], check=True)
    print(f"  saved a copy of the results so far to {dest}")


def _collect_and_report(dest: Path) -> None:
    run_dirs = sorted(m.parent for m in dest.glob("outputs/evals/*/*/metadata.json"))
    if not run_dirs:
        print(f"No vf-eval results came back; see {dest / 'logs' / 'run.log'}")
        return
    env_ids = set()
    for run_dir in run_dirs:
        env_ids.add(json.loads((run_dir / "metadata.json").read_text())["env_id"])
        subprocess.run([sys.executable, "tools/collect_run.py", str(run_dir)])
    for env_id in sorted(env_ids):
        subprocess.run([sys.executable, "tools/report_card.py", f"data/raw/{env_id}.jsonl"])


def _delete_pod(pod_id: str) -> None:
    done = subprocess.run(["runpodctl", "pod", "delete", pod_id], capture_output=True, text=True)
    if done.returncode == 0:
        print(f"Deleted pod {pod_id}. Billing for it has stopped.")
        STATE_FILE.unlink(missing_ok=True)
    else:
        print(f"⚠ Could not delete pod {pod_id}: {done.stderr.strip()}\n  Delete it yourself: runpodctl pod delete {pod_id}")


def _warn_about_running_pods() -> None:
    try:
        pods = _runpodctl("pod", "list")
    except LadderRunError as e:
        print(f"⚠ Could not check for leftover pods: {e}")
        return
    for pod in pods or []:
        print(f"⚠ Pod {pod.get('id')} is still on your account and may be billing: runpodctl pod delete {pod.get('id')}")


if __name__ == "__main__":
    main()
