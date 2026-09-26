from __future__ import annotations

import shlex

from hypothesis import given
from hypothesis import strategies as st

from churro.pod_ladder import (
    MAX_COMPLETION_TOKENS,
    OLLAMA_CONTEXT_LENGTH,
    UV_VERSION,
    GpuOffer,
    LadderPlan,
    LadderStatus,
    parallel_requests_for,
    parse_status,
    rank_gpu_offers,
    render_pod_script,
    vf_eval_command,
)
from churro.schema import ModelId

PARALLEL = 6
RTX_4090 = "NVIDIA GeForce RTX 4090"
RTX_3090 = "NVIDIA GeForce RTX 3090"
A100 = "NVIDIA A100 80GB PCIe"


def _gpu(
    gpu_id: str,
    *,
    community: float | None,
    secure: float | None,
    available: bool = True,
    memory_gb: int = 24,
) -> dict[str, object]:
    return {
        "gpuId": gpu_id,
        "memoryInGb": memory_gb,
        "available": available,
        "communityCloud": community is not None,
        "communityPricePerHr": community,
        "secureCloud": secure is not None,
        "securePricePerHr": secure,
    }


def _plan(ladder: tuple[str, ...] = ("qwen3:0.6b", "qwen3:14b")) -> LadderPlan:
    return LadderPlan(
        env_slug="liuliu/regex-craft",
        ladder=tuple(ModelId(m) for m in ladder),
        num_examples=30,
        rollouts_per_example=4,
        per_model_timeout_s=2700,
        per_rollout_timeout_s=600,
        self_destruct_s=9000,
        parallel=PARALLEL,
    )


class TestRankGpuOffers:
    def test_follows_preference_order_not_price_order(self) -> None:
        gpus = [
            _gpu(RTX_3090, community=0.22, secure=0.50),
            _gpu(RTX_4090, community=0.34, secure=0.74),
        ]
        preferences = [(RTX_4090, "COMMUNITY"), (RTX_3090, "COMMUNITY")]

        assert rank_gpu_offers(gpus, preferences, max_price_per_hr=1.0) == [
            GpuOffer(RTX_4090, "COMMUNITY", 0.34, 24),
            GpuOffer(RTX_3090, "COMMUNITY", 0.22, 24),
        ]

    def test_drops_offers_above_price_cap(self) -> None:
        gpus = [_gpu(RTX_4090, community=0.34, secure=0.74)]
        preferences = [(RTX_4090, "SECURE"), (RTX_4090, "COMMUNITY")]

        assert rank_gpu_offers(gpus, preferences, max_price_per_hr=0.5) == [
            GpuOffer(RTX_4090, "COMMUNITY", 0.34, 24)
        ]

    def test_offer_priced_exactly_at_cap_is_kept(self) -> None:
        cap = 0.34
        gpus = [_gpu(RTX_4090, community=cap, secure=None)]

        assert rank_gpu_offers(gpus, [(RTX_4090, "COMMUNITY")], cap) == [
            GpuOffer(RTX_4090, "COMMUNITY", cap, 24)
        ]

    def test_drops_cloud_tier_the_gpu_is_not_offered_on(self) -> None:
        gpus = [_gpu(RTX_4090, community=None, secure=0.74)]

        assert rank_gpu_offers(gpus, [(RTX_4090, "COMMUNITY")], 1.0) == []

    def test_drops_unavailable_gpus(self) -> None:
        gpus = [_gpu(RTX_4090, community=0.34, secure=0.74, available=False)]

        assert rank_gpu_offers(gpus, [(RTX_4090, "COMMUNITY")], 1.0) == []

    def test_ignores_gpus_not_in_preferences(self) -> None:
        gpus = [_gpu(A100, community=0.10, secure=0.20)]

        assert rank_gpu_offers(gpus, [(RTX_4090, "COMMUNITY")], 1.0) == []


class TestParallelRequestsFor:
    def test_48gb_card_runs_eight_requests_at_once(self) -> None:
        assert parallel_requests_for(48) == 8

    def test_24gb_card_keeps_four_so_14b_fits(self) -> None:
        assert parallel_requests_for(24) == 4

    def test_boundary_just_below_40gb_stays_at_four(self) -> None:
        assert parallel_requests_for(39) == 4

    def test_boundary_at_40gb_goes_to_eight(self) -> None:
        assert parallel_requests_for(40) == 8


class TestVfEvalCommand:
    def test_targets_local_ollama_with_plan_sizes_and_timeout(self) -> None:
        plan = _plan()
        model = ModelId("qwen3:1.7b")

        assert vf_eval_command(plan, model) == [
            "vf-eval",
            "regex-craft",
            "--model",
            "qwen3:1.7b",
            "--num-examples",
            "30",
            "--rollouts-per-example",
            "4",
            "--api-base-url",
            "http://127.0.0.1:11434/v1",
            "--api-key-var",
            "OPENAI_API_KEY",
            "--max-concurrent",
            str(PARALLEL),
            "--timeout",
            "600",
            "--max-tokens",
            str(MAX_COMPLETION_TOKENS),
            "--save-results",
            "--disable-tui",
            "--abbreviated-summary",
        ]


class TestRenderPodScript:
    def test_arms_self_destruct_before_installing_anything(self) -> None:
        script = render_pod_script(_plan())

        assert script.index("sleep 9000") < script.index("apt-get")

    def test_installs_the_plan_environment_from_its_hub_index(self) -> None:
        script = render_pod_script(_plan())

        assert (
            "uv pip install --python .venv/bin/python regex-craft "
            "--extra-index-url "
            "https://hub.primeintellect.ai/liuliu/regex-craft/install/simple/"
        ) in script

    def test_installs_pinned_uv_instead_of_trusting_the_image(self) -> None:
        script = render_pod_script(_plan())

        assert f"https://astral.sh/uv/{UV_VERSION}/install.sh" in script

    def test_does_not_depend_on_the_prime_cli(self) -> None:
        script = render_pod_script(_plan())

        assert "prime env install" not in script

    def test_setup_only_succeeds_if_the_environment_loads(self) -> None:
        script = render_pod_script(_plan())
        setup_body = script[script.index("setup() {") : script.index("\n}\n")]

        assert 'vf.load_environment("regex-craft")' in setup_body

    def test_ollama_serves_as_many_requests_as_vf_eval_sends(self) -> None:
        script = render_pod_script(_plan())

        assert f"OLLAMA_NUM_PARALLEL={PARALLEL} " in script

    def test_ollama_context_fits_a_full_length_completion(self) -> None:
        script = render_pod_script(_plan())

        assert OLLAMA_CONTEXT_LENGTH > MAX_COMPLETION_TOKENS
        assert f"OLLAMA_CONTEXT_LENGTH={OLLAMA_CONTEXT_LENGTH} " in script

    def test_bounds_each_model_run_by_the_per_model_timeout(self) -> None:
        plan = _plan()
        script = render_pod_script(plan)

        for model in plan.ladder:
            wrapped = f"timeout 2700 {shlex.join(vf_eval_command(plan, model))}"
            assert wrapped in script

    def test_signals_completion_as_the_final_status_write(self) -> None:
        script = render_pod_script(_plan())

        last_status_write = script.rstrip().splitlines()[-1]
        assert last_status_write == "mark ALL done"

    @given(
        st.lists(
            st.sampled_from(
                ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
            ),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    def test_runs_models_in_ladder_order(self, ladder: list[str]) -> None:
        script = render_pod_script(_plan(tuple(ladder)))

        positions = [script.index(f"--model {m} ") for m in ladder]
        assert positions == sorted(positions)


class TestParseStatus:
    def test_empty_status_means_nothing_has_happened(self) -> None:
        assert parse_status("") == LadderStatus(steps={}, finished=False)

    def test_later_line_for_same_step_wins(self) -> None:
        text = "qwen3:0.6b\trunning\nqwen3:0.6b\tdone\n"

        assert parse_status(text) == LadderStatus(
            steps={"qwen3:0.6b": "done"}, finished=False
        )

    def test_all_done_marks_run_finished(self) -> None:
        text = "SETUP\tdone\nqwen3:0.6b\ttimeout\nALL\tdone\n"

        assert parse_status(text) == LadderStatus(
            steps={"SETUP": "done", "qwen3:0.6b": "timeout", "ALL": "done"},
            finished=True,
        )

    def test_setup_failure_marks_run_finished(self) -> None:
        text = "SETUP\tfailed\n"

        assert parse_status(text) == LadderStatus(
            steps={"SETUP": "failed"}, finished=True
        )

    def test_skips_blank_and_malformed_lines(self) -> None:
        text = "\nnot-a-status-line\nqwen3:4b\tdone\n"

        assert parse_status(text) == LadderStatus(
            steps={"qwen3:4b": "done"}, finished=False
        )
