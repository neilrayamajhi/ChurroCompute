from __future__ import annotations

import shlex

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from churro.schema import ModelId

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
# Caps qwen3 thinking loops; normal answers here stay well under it.
MAX_COMPLETION_TOKENS = 16384
# Prompt + a full-length completion must fit, or Ollama shifts the window and
# the model loses its own prompt, which is how runaway generations start.
OLLAMA_CONTEXT_LENGTH = 20480
POD_WORKDIR = "/root/churro"
UV_VERSION = "0.12.5"
PRIME_HUB_INDEX = "https://hub.primeintellect.ai/{slug}/install/simple/"

Cloud = Literal["COMMUNITY", "SECURE"]


@dataclass(frozen=True, slots=True)
class GpuOffer:
    gpu_id: str
    cloud: Cloud
    price_per_hr: float
    memory_gb: int


@dataclass(frozen=True, slots=True)
class LadderPlan:
    env_slug: str
    ladder: tuple[ModelId, ...]
    num_examples: int
    rollouts_per_example: int
    per_model_timeout_s: int
    per_rollout_timeout_s: int
    self_destruct_s: int
    parallel: int


@dataclass(frozen=True, slots=True)
class LadderStatus:
    steps: dict[str, str]
    finished: bool


def rank_gpu_offers(
    gpus: Sequence[dict[str, object]],
    preferences: Sequence[tuple[str, Cloud]],
    max_price_per_hr: float,
) -> list[GpuOffer]:
    by_id = {g["gpuId"]: g for g in gpus if g.get("available")}
    offers: list[GpuOffer] = []
    for gpu_id, cloud in preferences:
        gpu = by_id.get(gpu_id)
        if gpu is None:
            continue
        price = gpu.get(
            "communityPricePerHr" if cloud == "COMMUNITY" else "securePricePerHr"
        )
        if isinstance(price, int | float) and price <= max_price_per_hr:
            offers.append(GpuOffer(gpu_id, cloud, float(price), int(gpu["memoryInGb"])))
    return offers


def parallel_requests_for(memory_gb: int) -> int:
    # 14b weights (~9GB) plus a full-context KV cache per request fit 8 slots
    # on a 48GB card; on 24GB, 8 slots would spill to CPU and crawl.
    return 8 if memory_gb >= 40 else 4


def vf_eval_command(plan: LadderPlan, model: ModelId) -> list[str]:
    return [
        "vf-eval",
        plan.env_slug.rsplit("/", 1)[-1],
        "--model",
        model,
        "--num-examples",
        str(plan.num_examples),
        "--rollouts-per-example",
        str(plan.rollouts_per_example),
        "--api-base-url",
        OLLAMA_BASE_URL,
        "--api-key-var",
        "OPENAI_API_KEY",
        "--max-concurrent",
        str(plan.parallel),
        "--timeout",
        str(plan.per_rollout_timeout_s),
        "--max-tokens",
        str(MAX_COMPLETION_TOKENS),
        "--save-results",
        "--disable-tui",
        "--abbreviated-summary",
    ]


def render_pod_script(plan: LadderPlan) -> str:
    pulls = " && ".join(f"ollama pull {shlex.quote(m)}" for m in plan.ladder)
    env_name = plan.env_slug.rsplit("/", 1)[-1]
    load_check = f'import verifiers as vf; vf.load_environment("{env_name}")'
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"cd {POD_WORKDIR}",
        "mkdir -p logs",
        f'mark() {{ printf "%s\\t%s\\n" "$1" "$2" >> {POD_WORKDIR}/status.tsv; }}',
        "exec >> logs/run.log 2>&1",
        # SSH shells don't inherit the pod's env; RUNPOD_* lives on PID 1.
        "while IFS= read -r -d '' kv; do case \"$kv\" in RUNPOD_*) export \"$kv\";; esac; done < /proc/1/environ",
        'if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then',
        f'  nohup setsid bash -c "sleep {plan.self_destruct_s}; runpodctl remove pod $RUNPOD_POD_ID || runpodctl pod delete $RUNPOD_POD_ID" >/dev/null 2>&1 < /dev/null &',
        "  mark SELF_DESTRUCT armed",
        "else",
        "  mark SELF_DESTRUCT unavailable",
        "fi",
        'export PATH="$HOME/.local/bin:$PATH" DEBIAN_FRONTEND=noninteractive OPENAI_API_KEY=ollama',
        "setup() {",
        "  apt-get update -qq && apt-get install -y -qq lshw zstd curl &&",
        "  curl -fsSL https://ollama.com/install.sh | sh &&",
        f"  (nohup setsid env OLLAMA_HOST=127.0.0.1 OLLAMA_NUM_PARALLEL={plan.parallel} OLLAMA_CONTEXT_LENGTH={OLLAMA_CONTEXT_LENGTH} OLLAMA_KEEP_ALIVE=30m ollama serve > logs/ollama.log 2>&1 < /dev/null &) &&",
        # Don't trust whatever uv the image happens to ship.
        f"  curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh &&",
        "  uv sync --frozen --no-dev &&",
        # Same install `prime env install` performs, minus the prime CLI's own
        # dependency tree, which clashes with verifiers 0.3.0 on Linux.
        f"  uv pip install --python .venv/bin/python {shlex.quote(env_name)} --extra-index-url {PRIME_HUB_INDEX.format(slug=plan.env_slug)} &&",
        # Loading usually downloads the dataset from Hugging Face, which
        # rate-limits anonymous requests from shared RunPod IPs (429): retry.
        f"  {{ .venv/bin/python -c {shlex.quote(load_check)} ||"
        f" {{ sleep 120; .venv/bin/python -c {shlex.quote(load_check)}; }} ||"
        f" {{ sleep 300; .venv/bin/python -c {shlex.quote(load_check)}; }}; }} &&",
        "  for _ in $(seq 60); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 2; done &&",
        f"  {pulls}",
        "}",
        "mark SETUP running",
        "if setup; then mark SETUP done; else mark SETUP failed; exit 1; fi",
        "source .venv/bin/activate",
    ]
    for model in plan.ladder:
        run = shlex.join(vf_eval_command(plan, model))
        lines += [
            f"mark {model} running",
            f"timeout {plan.per_model_timeout_s} {run}",
            "rc=$?",
            f'if [ $rc -eq 0 ]; then mark {model} done; elif [ $rc -eq 124 ]; then mark {model} timeout; else mark {model} failed; fi',
        ]
    lines.append("mark ALL done")
    return "\n".join(lines) + "\n"


def parse_status(text: str) -> LadderStatus:
    steps: dict[str, str] = {}
    for line in text.splitlines():
        step, sep, state = line.partition("\t")
        if sep and step and state:
            steps[step] = state
    finished = steps.get("ALL") == "done" or steps.get("SETUP") == "failed"
    return LadderStatus(steps=steps, finished=finished)
