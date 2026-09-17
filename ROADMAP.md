# Churro Roadmap

Where the project stands, what ships next, and what the PRD doesn't cover.

Legend: `[x]` done · `[~]` in progress · `[ ]` not started

---

## You Are Here

**Phase 1 complete.** The full pipe runs end-to-end on the M4:
uv + Ollama + qwen3:1.7b + verifiers + prime CLI + a real environment
(`primeintellect/gsm8k`) all working. First 6 rollouts scored `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` —
which itself demonstrates the exact problem Churro exists to detect
(zero-signal saturation, per PRD Part 2).

**Next action:** Phase 2 — inspect vf-eval output, write `collect.py`,
normalize to Churro's own schema. Blocking everything downstream.

---

## Phase 0 — Environment Setup [x]

*Reference: PRD Part 4.*

- [x] Homebrew, uv, Ollama installed
- [x] `qwen3:1.7b` pulled (1.4GB)
- [x] `qwen3:8b` pulled (5.2GB) — worked first try in parallel batch pull 2026-09-02 after 3 prior serial failures
- [x] `qwen3:0.6b` (522MB), `qwen3:4b` (2.5GB), `qwen3:14b` (9.3GB) pulled 2026-09-02
- [x] `uv init churro`, Python 3.12 venv, `verifiers 0.3.0`
- [x] `prime` CLI installed + logged in as Neil Rayamajhi / team Churro
- [x] `.env` with `OPENAI_BASE_URL=http://localhost:11434/v1`
- [x] `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=30m` in `~/.zshrc`

**Exit criterion (met):** `vf-eval` returns non-null rewards.

---

## Phase 1 — First Rollout [x]

*Reference: PRD Part 5.*

- [x] Install a simple environment (`primeintellect/gsm8k` — GSM8K math)
- [x] Run `vf-eval gsm8k -m qwen3:1.7b -n 3 -r 2 -s` successfully
- [x] Confirm scores appear in `outputs/evals/gsm8k--qwen3:1.7b/<hash>/results.jsonl`

**Exit criterion (met):** stack works end-to-end.

---

## Phase 2 — Data Normalization [x]

*Reference: PRD Part 6. Estimated: ~½ day.*

Do not skip. Every metric downstream reads from this schema, not from
vf-eval's raw output. One library update should break one file, not six.

- [ ] Write `tools/inspect_output.py` — dumps one record so you can eyeball fields
- [ ] Confirm the reward field name (probably `reward`) and task-id field
- [ ] Write `churro/collect.py`:
  - reads raw vf-eval JSON/JSONL
  - emits one JSON object per rollout to `data/raw/<env_id>.jsonl`
  - required keys: `env_id, task_id, group_id, rollout_idx, model, reward, num_turns, completion_tokens, collected_at, churro_version`
  - `group_id = "{env_id}::{task_id}::{model}"` — critical for reconstructing GRPO groups
- [ ] Run it against the gsm8k rollout — verify 6 rows out

**Exit criterion:** every future metric reads `data/raw/*.jsonl` exclusively.

---

## Phase 3 — Signal Rate (Metric 1 of 6) [x]

*Reference: PRD Part 7. Estimated: ~1 day.*

The headline metric — everything else is supporting evidence.

- [ ] Collect real data: `vf-eval gsm8k -m qwen3:1.7b -n 30 -r 8 -s`
  (30 tasks × 8 attempts = GRPO group size)
- [ ] Normalize into `data/raw/gsm8k.jsonl`
- [ ] Write `churro/metrics/signal.py`:
  - group by `group_id`
  - live groups = attempts disagree (`len(set(scores)) > 1`)
  - report `signal_rate`, 95% CI, `dead_too_easy`, `dead_too_hard`, `mean_spread`
- [ ] Run against gsm8k — expected: near-zero signal (model too strong for grade-school math)
- [ ] Run against 2 more environments — compare, prove the metric discriminates

**Exit criterion:** signal-rate numbers with error bars for ≥3 environments,
with the model + group size stamped on every result.

---

## Phase 4 — Difficulty Curve (Metric 2 of 6) [ ]

*Reference: PRD Part 8. Estimated: ~2 days.*

Difficulty is a relationship between task and model, so measure across a model ladder.

- [ ] Freeze the ladder in `churro/config.py`:
  `LADDER = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]`
  `LADDER_VERSION = "qwen3-v1"`
- [ ] Pull all 5 models (blocked until 8b + 14b download; 14b tight on 16GB RAM)
- [ ] Collect: loop over ladder, `vf-eval` each × 30 tasks × 8 attempts
- [ ] Write `churro/metrics/difficulty.py`:
  - pass_rate + mean_reward + signal_rate per model
  - report `floor`, `ceiling`, `slope`, `saturated`, `possibly_broken`, `best_signal_model`
- [ ] Run on ≥3 environments

**Decision point:** if 14B won't fit on 16GB RAM comfortably, either
skip it and version the ladder as `qwen3-m4-16gb-v1`, or rent a Mac Studio hour.

---

## Phase 5 — Metrics 3–6 [ ]

*Reference: PRD Parts 9–12. Estimated: ~1 week combined.*

- [ ] **Consistency** (Part 9): re-score fixed rollouts N times, check for score drift.
      LLM-judge graders will jitter, programmatic graders shouldn't. Any nonzero
      value on a "deterministic" grader is a real bug worth reporting.
- [ ] **Reward Shape** (Part 10): entropy-based `effective_bins` on the score
      distribution. Catches "graded" rewards that are effectively binary.
- [ ] **Exploitability** (Part 11): read `djinn` first — do NOT invent exploit
      categories from scratch. HUD ships a live product here — weigh how much
      time this deserves.
- [ ] **Grader Narrowness** (Part 12): false-negative rate on alternate valid
      solutions. Hardest metric — depends on curating known-correct alternates.

**Exit criterion:** all six metrics runnable on any conforming environment.

---

## Phase 6 — Package as CLI [ ]

*Reference: PRD Part 13. Estimated: ~3 days.*

- [ ] Restructure to PRD's target layout (`churro/{config,collect,fingerprint,report,cli}.py`, `churro/metrics/`)
- [ ] `churro/fingerprint.py` — runs all metrics, writes one `data/fingerprints/<env>.json`
- [ ] `churro/report.py` — JSON → readable card
- [ ] `churro/cli.py` — `churro run <env>` command
- [ ] Wire the entry point in `pyproject.toml` (`[project.scripts]` — already there)
- [ ] Test: `uv run churro run gsm8k` produces a full report card

**Four non-negotiables per PRD:**
1. Cache rollouts. Never re-run inference to recompute a metric.
2. Version everything (ladder, group size, sample count, env version, date, churro version).
3. Error bars on every number.
4. Fail loudly. Never emit a fingerprint of zeros.

---

## Phase 7 — Fingerprint Everything [ ]

*Reference: PRD Part 14. Estimated: ~2 weeks (mostly wall-clock waiting).*

- [ ] Estimate: time one env end-to-end, multiply by 200, decide viability
- [ ] Adjust `n_tasks` down if the total exceeds a week of overnight runs
- [ ] Loop over Hub environments (start with `--owner primeintellect`)
- [ ] Log every failure: broken packages, missing deps, Docker requirements,
      external API keys. Failure rate is itself a finding.
- [ ] Machine must stay awake: plug in, disable sleep
- [ ] Publish the fingerprint dataset alongside the methodology

**This is the deliverable that gets noticed.** It's what nobody else has.

---

## Phase 8 — The Training Arm [ ]

*Reference: PRD Part 15. **Do not touch until Phase 7 is published.***

- [ ] One end-to-end training run on one env with `prime-rl` (measure real GPU-hour cost)
- [ ] Noise floor: 10 envs × 3 seeds. If seed variance swamps env variance, **stop and publish that** — it's a real finding that saves ~4 months
- [ ] Main sweep: 40 envs, stratified by signal-rate
- [ ] Transfer check: same envs, larger model, subset
- [ ] Pre-registered analysis, written before the sweep runs

**Cost:** ~$10K at $2–3/GPU-hour. This is the only paid step.

---

# Beyond the PRD

The PRD ends at Phase 8. The below is what the PRD does not cover but that real
users / adoption would eventually require. Order matters — later items assume earlier ones.

## B1 — Distribution [ ]

- [ ] Publish `churro` to PyPI (`uv build && uv publish`)
- [ ] Add `installation` section to README with `pip install churro-metrics`
      (or similar — pick a PyPI name that isn't taken; verify before committing)
- [ ] Version + tag releases (`git tag v0.1.0`, `gh release create`)
- [ ] Semantic versioning: metric-breaking changes → major bump

## B2 — Reproducibility Guarantees [ ]

- [ ] `churro/config.py` records ladder + versions in every fingerprint (PRD requires it — enforce in code)
- [ ] Ship a `requirements.lock` alongside `pyproject.toml` for exact-version repro
- [ ] Docker image (optional) so users on non-Mac hardware can reproduce your ladder
- [ ] Publish evaluation methodology as a citeable artifact (Zenodo DOI or arXiv preprint)

## B3 — Public Results Surface [ ]

The dataset in JSON is not the deliverable that goes viral. A browsable page is.

- [ ] Static site (Next.js + shadcn or Astro) reading fingerprints from your GitHub repo
- [ ] One page per environment: signal rate, difficulty curve chart, grader consistency, shape histogram
- [ ] Leaderboard: worst-signal, most-saturated, most-inconsistent
- [ ] Deploy to Vercel or GitHub Pages
- [ ] `churrocompute.dev` or similar domain (optional)

## B4 — Continuous Fingerprinting [ ]

- [ ] Nightly GitHub Actions cron: re-fingerprint recently-updated Hub environments
- [ ] Alert (email / GitHub issue) when an env's signal rate drops sharply — indicates the env was updated
- [ ] Public "last fingerprinted" timestamp per env on the site

## B5 — Community & Adoption [ ]

- [ ] README with 3 example fingerprints + "run this on your own env in 5 min"
- [ ] Blog post: "Here's the signal rate on 200 published RL environments" — this is the launch
- [ ] Submit to `ML Twitter/X`, HN, LessWrong, Prime Intellect Discord
- [ ] File issues on the worst-scoring envs' repos with the fingerprint report attached
- [ ] Prime Intellect / Hub integration: PR to display Churro fingerprints inline on env pages

## B6 — Feedback Loop with Vendors [ ]

- [ ] Contact the ≥3 largest env-selling companies. Offer to run Churro on their portfolio in exchange for citation
- [ ] Track: does publishing a fingerprint move the vendor to fix their env? That's a stat worth logging
- [ ] Consider a "Churro Verified" badge (later — only after methodology is respected)

## B7 — Business Model (optional, only if Phase 8 validates) [ ]

Only meaningful if training-arm results in Phase 8 prove Churro's metrics
actually predict training outcomes. If they don't, Churro is still an
academic contribution — but the below is off the table.

- [ ] Private fingerprinting service for labs: run Churro on their internal envs
- [ ] Pricing model: per-env fingerprint (e.g. $X per 30-task × 8-rollout run)
- [ ] Anthropic / OpenAI / xAI as customers before broader launch
- [ ] Enterprise vs. OSS split — keep the metrics OSS, sell the compute + curation

## B8 — Long-Horizon Research [ ]

- [ ] Extend the ladder beyond Qwen3 (Llama, Mistral, Gemma) to test cross-family generalization
- [ ] Test the assumption that fingerprints computed on 30 tasks predict fingerprints on 300 tasks
- [ ] Adversarial section: red-team your own metrics (can we construct an env that scores perfectly on all six but trains models poorly?)

---

# Critical Path

If forced to a single line: **Phase 2 → Phase 3 → Phase 7 → B3 (public site) → Phase 8**.

Everything else is optional. Metrics 3–6 (Phase 5) add credibility but Phase 3's
Signal Rate alone is enough to publish an interesting dataset. The public site
(B3) matters more for adoption than any additional metric would.

---

# Non-Negotiables

Copied out of the PRD because they're easy to forget:

1. **Cache rollouts.** Never re-run inference to recompute a metric.
2. **Version everything.** Ladder version, group size, sample count, env version, date, churro version — every fingerprint carries these.
3. **Error bars on every number.** Bare point estimates undercut the whole product's premise.
4. **Fail loudly.** If an env won't build, say so. Never emit a fingerprint of zeros.
5. **Do not touch Phase 8 until Phase 7 is published.** It's a $10K + months commitment.
